# -*- coding: utf-8 -*-
from helpers import *
from parameters import *
from task_evaluator import TaskEvaluator
from cameravision import *
from peas.networks.rnn import NeuralNetwork
from threading import Condition, Lock
from socket import error as socket_error

import gobject
import glib
import dbus
import dbus.mainloop.glib
from copy import deepcopy
import json
import time
import sys
import socket
import thread
import cv2
import math

EVALUATIONS = 1000
MAX_MOTOR_SPEED = 150
TIME_STEP = 0.005
ACTIVATION_FUNC = 'tanh'
POPSIZE = 30
GENERATIONS = 100
TARGET_SPECIES = 2
SOLVED_AT = EVALUATIONS * 2
EXPERIMENT_NAME = 'NEAT_foraging_task_distributed_large'

INITIAL_ENERGY = 750
MAX_ENERGY = 1500
ENERGY_DECAY = 5
MAX_STEPS = 200
EXPECTED_FPS = 4

PUCK_BONUS_SCALE = 1
GOAL_BONUS_SCALE = 2
GOAL_REACHED_BONUS = INITIAL_ENERGY
BACKWARD_PUNISH_SCALE = 10
TURN_AWAY_PUNISH_SCALE = 10
ANGLE_PENALTY = 3

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')

class ForagingTask(TaskEvaluator):

    def __init__(self, thymioController, commit_sha, debug=False):
        TaskEvaluator.__init__(self, thymioController, commit_sha, debug, EXPERIMENT_NAME, EVALUATIONS, TIME_STEP,
                               ACTIVATION_FUNC, POPSIZE, GENERATIONS, SOLVED_AT)
        self.camera = CameraVisionVectors(False, self.logger)
        self.ctrl_thread_started = False
        self.img_thread_started = False
        self.individuals_evaluated = 0
        self.presenceValues = []
        self.presence = []
        self.prev_presence = []
        self.conditionLock = Condition()
        self.conditionGoalReached = Condition()
        self.motorLock = Lock()
        self.presenceValuesReady = False
        self.proxvalues = [-1, -1]
        self.goalReached = False
        self.puckRemoved = False
        self.hasPuck = False
        self.puckLostCounter = 0
        self.goalReachedWaiter = 0
        self.goalReachedCounter = 0

    """
        The _step function will only be called when the values from the camera are ready to be consumed. It will not
        be called from the main thread and therefore requires locking of resources which could possibly be used by the
        main thread, like the thymio controller.
    """
    def _step(self, evaluee, callback):

        self.motorLock.acquire()
        getProxReadings(self.thymioController, self.prox_readings_ok_call, self.prox_readings_nok_call)
        time.sleep(0.5)
        self.motorLock.release()

        presence_box = self.presenceValues["puck"]
        presence_goal = self.presenceValues["target"]

        if presence_goal and presence_box:
            self.prev_presence = self.presence
            self.presence = tuple(presence_box) + tuple(presence_goal)

            has_box = 1 if presence_box[0] == 0 else 0

            #inputs consist of:
            #   2 values for distance and angle of the puck
            #   2 values for distance and angle of the target
            #   2 values for the proximity sensors
            #   boolean whether robot currently has the puck
            #   a constant value of 1 (bias)

            inputs = np.hstack(([x if not x == -np.inf else -self.camera.MAX_DISTANCE for x in self.presence]))
            inputs[::2] = map(lambda dist: dist/float(self.camera.MAX_DISTANCE), inputs[::2])
            inputs = np.concatenate((inputs, self.proxvalues), 0)
            inputs = np.hstack((inputs, has_box, 1))

            #print "Inputs: ", inputs

            out = NeuralNetwork(evaluee).feed(inputs)
            left, right = list(out[-2:])
            self.motorspeed = { 'left': left, 'right': right }
            self.motorLock.acquire()
            writeMotorSpeed(self.thymioController, self.motorspeed, max_speed=MAX_MOTOR_SPEED)
            self.motorLock.release()
        else:
            time.sleep(.001)

        callback(self.getEnergyDelta2())
        return True

    def prox_readings_ok_call(self, psvaleus):
        self.proxvalues = np.array([(psvaleus[0] + psvaleus[4])/2, (psvaleus[5] + psvaleus[6])/2] ,dtype='f')
        self.proxvalues = [(float(x) - float(SENSOR_MAX[0]/2))/float(SENSOR_MAX[0]/2) for x in self.proxvalues]

    def prox_readings_nok_call(self):
        print "Error reading proximity values"

    def getFitness(self):
        print "evaluations taken ", self.evaluations_taken, " energy: ", self.energy
        energy_norm = math.tanh(self.energy)
        return max(self.evaluations_taken + energy_norm, 1)

    def getEnergyDelta2(self):
        global img_client
        if img_client and self.camera.binary_channels is not None:
            try:
                send_image(img_client, self.camera.binary_channels, self.energy,
                           self.presence[0], self.presence[2])
            except socket_error:
                print "could not send image"

        self.presence = [x if not x == -np.inf else self.camera.MAX_DISTANCE for x in self.presence]
        self.prev_presence = [x if not x == -np.inf else self.camera.MAX_DISTANCE for x in self.prev_presence]

        if None in self.presence or None in self.prev_presence:
            return 0

        prox_penalty_front = (self.proxvalues[0] + 1) * 20
        prox_penalty_back = (self.proxvalues[1] + 1) * 10

        if self.hasPuck:
            # test to see if the robot turned away from the goal.
            prev_saw_goal = self.prev_presence[2] is not self.camera.MAX_DISTANCE and \
                self.presence[2] is self.camera.MAX_DISTANCE

            # use prev angle in calculation: -1 -> minimal influence distance, 1 -> maximum influence distance
            angle_goal_diff = abs(self.prev_presence[3]) - abs(self.presence[3])

            turn_away_punishment = 0
            if prev_saw_goal:
                turn_away_punishment = TURN_AWAY_PUNISH_SCALE
                print "punishing for turning away, angle difference = ", angle_goal_diff

            delta_goal_distance = (abs(self.prev_presence[2]) - abs(self.presence[2]))
            if abs(delta_goal_distance) > 60:
                delta_goal_distance = 10

            energy_delta = GOAL_BONUS_SCALE * delta_goal_distance + ANGLE_PENALTY * angle_goal_diff \
                           - turn_away_punishment

        else:
            delta_puck_distance = (abs(self.prev_presence[0]) - abs(self.presence[0]))
            angle_puck_diff = abs(self.prev_presence[1]) - abs(self.presence[1])

            prev_saw_puck = self.prev_presence[0] is not self.camera.MAX_DISTANCE and \
                            self.presence[0] is self.camera.MAX_DISTANCE

            turn_away_punishment = 0
            if prev_saw_puck:
                turn_away_punishment = TURN_AWAY_PUNISH_SCALE
                print "punishing for turning away, angle difference = ", angle_puck_diff

            if abs(delta_puck_distance) > 60:
                delta_puck_distance = 10
            if angle_puck_diff < 0:
                angle_puck_diff *= 2

            energy_delta = PUCK_BONUS_SCALE * delta_puck_distance + ANGLE_PENALTY * angle_puck_diff - turn_away_punishment

        energy_delta = energy_delta - prox_penalty_front - prox_penalty_back


        if self.presence[0] == 0:
            self.puckLostCounter = 0
            self.hasPuck = True
        elif self.hasPuck:
            self.puckLostCounter += 1
            if self.puckLostCounter == 10:
                self.puckLostCounter = 0
                self.hasPuck = False
        else:
            self.hasPuck = False

        if self.hasPuck:
            if self.camera.goal_reached(self.hasPuck, self.presence[2], MIN_GOAL_DIST):
                self.goalReachedWaiter += 1
                energy_delta = self.checkForGoal()

            else:
                if not self.goalReachedWaiter == 0:
                    self.goalReachedWaiter -= 1


        print "E delta: %.2f\t" % energy_delta, "P goal: %.2f\t" % self.presence[2], "P puck: %.2f\t" %self.presence[0], "Puck: \t", self.hasPuck, "Goals: \t", self.goalReachedCounter

        #" prox penalties: ", [prox_penalty_front, prox_penalty_back], " goals reached", self.goalReachedCounter

        return energy_delta

    def goal_reach_camera_callback(self, presence):
        print("new presence values goal reached callback: ", presence)
        self.puckRemoved = presence["puck"][0] != 0
        if self.puckRemoved:
            self.conditionGoalReached.acquire()
            self.conditionGoalReached.notify()
            self.conditionGoalReached.release()

        self.conditionLock.acquire()
        self.presenceValuesReady = True
        self.conditionLock.notify()
        self.conditionLock.release()

    def checkForGoal(self):
        if self.goalReachedWaiter == 3:
            self.goalReached = True
            self.goalReachedWaiter = 0
            self.conditionLock.acquire()
            self.presenceValuesReady = True
            self.conditionLock.notify()
            self.conditionLock.release()
            self.hasPuck = False
            return GOAL_REACHED_BONUS
        else:
            return 0

    def cameraCallback(self, evaluee, callback, presenceValues):
        #activate the main thread waiting for camera input
        self.conditionLock.acquire()
        self.presenceValuesReady = True
        self.conditionLock.notify()
        self.conditionLock.release()

        # print "new presenceValues: ", presenceValues

        self.presenceValues = presenceValues
        self._step(evaluee, callback)
        self.evaluations_taken += 1
        self.energy -= ENERGY_DECAY

    def cameraErrorCallback(self):
        print "Camera Error occured"

    def cameraWait(self):
        self.conditionLock.acquire()
        self.presenceValuesReady = False
        while not self.presenceValuesReady:
            self.conditionLock.wait(0.2)
        self.conditionLock.release()

    def evaluate(self, evaluee):
        self.frame_rate_counter = 0
        self.step_time = time.time()

        global ctrl_client
        if ctrl_client and not self.ctrl_thread_started:
            thread.start_new_thread(check_stop, (self, ))
            self.ctrl_thread_started = True

        self.evaluations_taken = 0
        self.energy = INITIAL_ENERGY
        self.fitness = 0
        self.presence = self.prev_presence = (None, None)
        gobject.threads_init()
        dbus.mainloop.glib.threads_init()
        self.loop = gobject.MainLoop()

        def update_energy(task, energy):
            task.energy += energy

        def main_lambda(task):
            if task.energy <= 0 or task.evaluations_taken >= MAX_STEPS:
                task.motorLock.acquire()
                stopThymio(thymioController)
                task.motorLock.release()
                task.loop.quit()

                if task.energy <= 0:
                    print 'Energy exhausted'
                else:
                    print 'Time exhausted'

                return False

            if not self.goalReached:
                callback = lambda (psvalues): task.cameraCallback(evaluee, lambda (energy): update_energy(task, energy),
                                                                  psvalues)
            else:
                callback = self.goal_reach_camera_callback
                self.camera.update_callback(callback)
                print '===== Goal reached!'

                self.goalReachedCounter += 1

                self.motorLock.acquire()
                stopThymio(self.thymioController)
                self.motorLock.release()

                self.camera.update_callback(self.goal_reach_camera_callback)

                self.conditionGoalReached.acquire()
                self.puckRemoved = False
                while not self.puckRemoved:
                    self.thymioController.SendEventName('PlayFreq', [700, 0], reply_handler=dbusReply, error_handler=dbusError)
                    time.sleep(.3)
                    self.thymioController.SendEventName('PlayFreq', [0, -1], reply_handler=dbusReply, error_handler=dbusError)
                    self.conditionGoalReached.wait(0.7)

                self.conditionGoalReached.release()
                time.sleep(15)
                print "finished puck wait loop"
                self.goalReached = False
                self.prev_presence = list(self.prev_presence)
                if len(self.prev_presence) >= 3:
                    self.prev_presence[0] = self.prev_presence[2] = self.camera.MAX_DISTANCE
                time.sleep(2)


            if not self.camera.isAlive():
                print "starting camera"
                #Call the camera asynchroniously. Call the callback when the presence values are ready.
                self.camera.start_camera(callback, task.cameraErrorCallback)
                print "camera started"
            else:
                self.camera.update_callback(callback)
            self.cameraWait()
            return True
        gobject.timeout_add(int(self.timeStep * 1000), lambda: main_lambda(self))


        print 'Starting loop...'
        self.loop.run()

        fitness = self.getFitness()
        print 'Fitness at end: %d' % fitness

        if self.camera.isAlive():
            # self.camera.stop()
            self.camera.pause()
            # self.camera.join()

        self.motorLock.acquire()
        stopThymio(self.thymioController)
        self.motorLock.release()

        time.sleep(1)

        self.individuals_evaluated += 1

        return {'fitness': fitness}


def check_stop(task):
    global ctrl_client
    f = ctrl_client.makefile()
    line = f.readline()
    if line.startswith('stop'):
        task.motorLock.acquire()
        release_resources(task.thymioController)
        task.motorLock.release()
        task.exit(0)
    task.ctrl_thread_started = False

def release_resources(thymio):
    global ctrl_serversocket
    global ctrl_client
    ctrl_serversocket.close()
    if ctrl_client: ctrl_client.close()

    global img_serversocket
    global img_client
    img_serversocket.close()
    if img_client: img_client.close()

    stopThymio(thymio)

def write_header(client, boundary='thymio'):
    client.send("HTTP/1.0 200 OK\r\n" +
                "Connection: close\r\n" +
                "Max-Age: 0\r\n" +
                "Expires: 0\r\n" +
                "Cache-Control: no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0\r\n" +
                "Pragma: no-cache\r\n" +
                "Content-Type: multipart/x-mixed-replace; " +
                "boundary=" + boundary + "\r\n" +
                "\r\n" +
                "--" + boundary + "\r\n")


def send_image(client, binary_channels, energy, box_dist, goal_dist, boundary='thymio'):
    red = np.zeros(binary_channels[0].shape, np.uint8)
    cv2.putText(red, 'E: {0:.2f} P: {1:.0f} G: {2:.0f}'.format(energy, box_dist, goal_dist), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, ), 1, 255)
    image = np.dstack(binary_channels + [red])
    _, encoded = cv2.imencode('.png', image)
    image_bytes = bytearray(np.asarray(encoded))
    client.send("Content-type: image/png\r\n")
    client.send("Content-Length: %d\r\n\r\n" % len(image_bytes))
    client.send(image_bytes)
    client.send("\r\n--" + boundary + "\r\n")


if __name__ == '__main__':
    from peas.methods.odneat import NEATPopulation, NEATGenotype
    ctrl_ip = sys.argv[-2]
    genotype = lambda: NEATGenotype(
        inputs=8,
        outputs=2,
        types=[ACTIVATION_FUNC],
        prob_add_node=0.1,
        weight_range=(-3, 3),
        stdev_mutate_weight=.25,
        stdev_mutate_bias=.25,
        stdev_mutate_response=.25)
        #feedforward=False)
    pop = NEATPopulation(ctrl_ip, genotype, popsize=POPSIZE, target_species=TARGET_SPECIES, stagnation_age=5)

    log = { 'neat': {}, 'generations': [] }

    # log neat settings
    dummy_individual = genotype()
    log['neat'] = {
        'max_speed': MAX_MOTOR_SPEED,
        'evaluations': EVALUATIONS,
        'activation_function': ACTIVATION_FUNC,
        'popsize': POPSIZE,
        'generations': GENERATIONS,
        'initial_energy': INITIAL_ENERGY,
        'max_energy': MAX_ENERGY,
        'energy_decay': ENERGY_DECAY,
        'max_steps': MAX_STEPS,
        'puck_bonus_scale': PUCK_BONUS_SCALE,
        'goal_bonus_scale': GOAL_BONUS_SCALE,
        'goal_reached_bonus': GOAL_REACHED_BONUS,
        'elitism': pop.elitism,
        'tournament_selection_k': pop.tournament_selection_k,
        'target_species': pop.target_species,
        'stagnation_age': pop.stagnation_age,
        'feedforward': dummy_individual.feedforward,
        'initial_weight_stdev': dummy_individual.initial_weight_stdev,
        'prob_add_node': dummy_individual.prob_add_node,
        'prob_add_conn': dummy_individual.prob_add_conn,
        'prob_mutate_weight': dummy_individual.prob_mutate_weight,
        'prob_reset_weight': dummy_individual.prob_reset_weight,
        'prob_reenable_conn': dummy_individual.prob_reenable_conn,
        'prob_disable_conn': dummy_individual.prob_disable_conn,
        'prob_reenable_parent': dummy_individual.prob_reenable_parent,
        'stdev_mutate_weight': dummy_individual.stdev_mutate_weight,
        'stdev_mutate_bias': dummy_individual.stdev_mutate_bias,
        'stdev_mutate_response': dummy_individual.stdev_mutate_response,
        'weight_range': dummy_individual.weight_range
    }

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()
    thymioController = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'), dbus_interface='ch.epfl.mobots.AsebaNetwork')
    thymioController.LoadScripts(AESL_PATH, reply_handler=dbusReply, error_handler=dbusError)

    # switch thymio LEDs off
    thymioController.SendEventName('SetColor', [0, 0, 0, 0], reply_handler=dbusReply, error_handler=dbusError)

    debug = True
    commit_sha = sys.argv[-1]
    task = ForagingTask(thymioController, commit_sha, debug)

    ctrl_serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ctrl_serversocket.bind((sys.argv[-2], 1337))
    ctrl_serversocket.listen(5)
    ctrl_client = None
    def set_client():
        global ctrl_client
        print 'Control server: waiting for socket connections...'
        (ctrl_client, address) = ctrl_serversocket.accept()
        print 'Control server: got connection from', address
    thread.start_new_thread(set_client, ())

    img_serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    img_serversocket.bind((sys.argv[-2], 31337))
    img_serversocket.listen(5)
    img_client = None
    def set_img_client():
        global img_client
        print 'Image server: waiting for socket connections.'
        (img_client, address) = img_serversocket.accept()
        print 'Image server: got connection from', address
        write_header(img_client)
    thread.start_new_thread(set_img_client, ())

    def epoch_callback(population):
        # update log
        population_backup = population.giveBackUp()
        species_backup = population.giveBackUpSpecies()
        generation = { 'individuals': [], 'gen_number': population.generation }
        for individual in population_backup:
            copied_connections = { str(key): value for key, value in individual.conn_genes.items() }
            generation['individuals'].append({
                'node_genes': deepcopy(individual.node_genes),
                'conn_genes': copied_connections,
                'stats': deepcopy(individual.stats)
            })
        champion_file = task.experimentName + '_{}_{}.p'.format(commit_sha, population.generation)
        generation['champion_file'] = champion_file
        generation['species'] = [len(species.members) for species in species_backup]
        print generation['species']
        log['generations'].append(generation)

        task.getLogger().info(', '.join([str(ind.stats['fitness']) for ind in population_backup]))
        task.getLogger().info('goal reached counter: ', task.goalReachedCounter)
        jsonLog = open(task.jsonLogFilename, "w")
        json.dump(log, jsonLog)
        jsonLog.close()

        current_champ = population_backup.champions[-1]
        # print 'Champion: ' + str(current_champ.get_network_data())
        # current_champ.visualize(os.path.join(CURRENT_FILE_PATH, 'img/' + task.experimentName + '_%d.jpg' % population.generation))
        pickle.dump(current_champ, file(os.path.join(PICKLED_DIR, champion_file), 'w'))

    try:
        pop.epoch(generations=GENERATIONS, evaluator=task, solution=task, callback=epoch_callback)
    except KeyboardInterrupt:
        release_resources(task.thymioController)
        sys.exit(1)

    release_resources(task.thymioController)
    sys.exit(0)
