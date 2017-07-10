# -*- coding: utf-8 -*-

from parameters import *
# from neat_task import NEATTask

import numpy as np
import dbus
import dbus.mainloop.glib
import logging
logging.basicConfig()
import parameters as pr
from helpers import *
from task_evaluator import TaskEvaluator
from peas.networks.rnn import NeuralNetwork
import thread
import socket
from copy import deepcopy
import json
import sys

EVALUATIONS = 1000
TIME_STEP = 0.005
ACTIVATION_FUNC = 'tanh'
POPSIZE = 10
GENERATIONS = 30
TARGET_SPECIES = 2
SOLVED_AT = EVALUATIONS * 2
EXPERIMENT_NAME = 'NEAT_correct_logging'

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
MAIN_LOG_PATH = os.path.join(CURRENT_FILE_PATH, 'log_main')
OUTPUT_PATH = os.path.join(CURRENT_FILE_PATH, 'output')
PICKLED_DIR = os.path.join(CURRENT_FILE_PATH, 'pickled')
FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')


class ObstacleAvoidance(TaskEvaluator):

    def __init__(self, thymioController, commit_sha, debug=False, experimentName=EXPERIMENT_NAME, evaluations=EVALUATIONS, timeStep=0.005, activationFunction='tanh', popSize=POPSIZE, generations=GENERATIONS, solvedAt=1000):
        TaskEvaluator.__init__(self, thymioController, commit_sha, debug, experimentName, evaluations, timeStep, activationFunction, popSize, generations, solvedAt)
        self.ctrl_thread_started = False
        self.hitWallCounter = 0
        self.atWall = False
        print "New obstacle avoidance task"

    def evaluate(self, evaluee):
        global ctrl_client
        #print "thread started:", self.ctrl_thread_started
        if ctrl_client and not self.ctrl_thread_started:
            thread.start_new_thread(check_stop, (self, ))
            self.ctrl_thread_started = True

        return TaskEvaluator.evaluate(self, evaluee)
            
    def _step(self, evaluee, callback):
        def ok_call(psValues):
            psValues = np.array([psValues[0], psValues[2], psValues[4], psValues[5], psValues[6], 1],dtype='f')
            psValues[0:5] = [(float(x) - float(pr.SENSOR_MAX[0]/2))/float(pr.SENSOR_MAX[0]/2) for x in psValues[0:5]]
            left, right = list(NeuralNetwork(evaluee).feed(psValues)[-2:])
            motorspeed = { 'left': left, 'right': right }
            try:
                writeMotorSpeed(self.thymioController, motorspeed)
            except Exception as e:
                print str(e)

            # print "Sensor values: ", psValues, " sensor max: ", SENSOR_MAX
            if not self.atWall and any(i >= 1 for i in psValues[0:5]):
                self.atWall = True
                self.hitWallCounter += 1
            elif not any(i >= 1 for i in psValues[0:5]):
                self.atWall = False

            callback(self.getFitness(motorspeed, psValues))

        def nok_call():
            print " Error while reading proximity sensors"

        getProxReadings(self.thymioController, ok_call, nok_call)
        return True

    def getFitness(self, motorspeed, observation):
        # Calculate penalty for rotating
        # speedpenalty = 0
        # if motorspeed['left'] > motorspeed['right']:
        #     speedpenalty = float((motorspeed['left'] - motorspeed['right']))
        # else:
        #     speedpenalty = float((motorspeed['right'] - motorspeed['left']))

        speedpenalty = float(abs(motorspeed['left'] - motorspeed['right']))

        # Calculate normalized distance to the nearest object
        sensorpenalty = 0
        for i, sensor in enumerate(observation[:-1]):
            distance = sensor
            if sensorpenalty < distance:
                sensorpenalty = distance


        # fitness for 1 timestep in [-2, 2]
        return float(motorspeed['left'] + motorspeed['right']) * (1 - min(speedpenalty,1)) * (1 - min(sensorpenalty,1))


def check_stop(task):
    global ctrl_client
    f = ctrl_client.makefile()
    line = f.readline()
    if line.startswith('stop'):
        print "stopping"
        release_resources(task.thymioController)
        task.exit(0)
        task.loop.quit()
        sys.exit(1)
    task.ctrl_thread_started = False

def release_resources(thymio):
    global ctrl_serversocket
    global ctrl_client
    ctrl_serversocket.close()
    if ctrl_client: ctrl_client.close()
    
    stopThymio(thymio)


if __name__ == '__main__':
    from peas.methods.neat import NEATPopulation, NEATGenotype
    genotype = lambda: NEATGenotype(
        inputs=6, outputs=2,
        types=[ACTIVATION_FUNC],
        prob_add_node=0.1, 
        weight_range=(-3, 3),
        stdev_mutate_weight=.25,
        stdev_mutate_bias=.25,
        stdev_mutate_response=.25)
    pop = NEATPopulation(genotype, popsize=POPSIZE, target_species=TARGET_SPECIES)

    # log neat settings
    log = { 'neat': {}, 'generations': [] }
    dummy_individual = genotype()
    log['neat'] = {
        'max_speed': MAX_MOTOR_SPEED,
        'evaluations': EVALUATIONS,
        'activation_function': ACTIVATION_FUNC,
        'popsize': POPSIZE,
        'generations': GENERATIONS,
        'elitism': pop.elitism,
        'tournament_selection_k': pop.tournament_selection_k,
        'target_species': pop.target_species,
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

    debug = False
    commit_sha = sys.argv[-1]
    task = ObstacleAvoidance(thymioController, commit_sha, debug, EXPERIMENT_NAME)

    ctrl_serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ctrl_ip = sys.argv[-2]
    ctrl_serversocket.bind((ctrl_ip, 1337))
    ctrl_serversocket.listen(5)
    ctrl_client = None
    def set_client():
        global ctrl_client
        print 'Control server: waiting for socket connections...'
        (ctrl_client, address) = ctrl_serversocket.accept()
        print 'Control server: got connection from', address
    thread.start_new_thread(set_client, ())
    
    def epoch_callback(population):
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
        
        
        #champion_file = task.experimentName + '_{}_{}.p'.format(commit_sha, population.generation)
        #generation['champion_file'] = champion_file
        generation['species'] = [len(species.members) for species in species_backup]
        log['generations'].append(generation)
        
        print "fitnesses in obs avoidance %s" % [ind.stats['fitness'] for ind in population_backup]

        task.getLogger().info(', '.join([str(ind.stats['fitness']) for ind in population_backup]))
        jsonLog = open(task.jsonLogFilename, "w")
        json.dump(log, jsonLog)
        jsonLog.close()

    try:
        pop.epoch(generations=GENERATIONS, evaluator=task, callback=epoch_callback)
    except KeyboardInterrupt:
        release_resources(task.thymioController)
        sys.exit(1)

    release_resources(task.thymioController)
    sys.exit(0)
