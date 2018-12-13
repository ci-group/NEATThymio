# -*- coding: utf-8 -*-
from parameters import *
# from neat_task import NEATTask

import numpy as np
import dbus
import dbus.mainloop.glib
import logging
logging.basicConfig()
import parameters as pr
import object_detector
import cameracontroller
from thymio_task_evaluate import TaskEvaluator
from peas.networks.rnn import NeuralNetwork
import object_detector
import thread
import socket
import time
from copy import deepcopy
import json
import sys
import thymio_task
import math
from helpers import *


#### BEGIN OF PARAMETER SETTING ########################

# Set parameters: Foraging task related and network
time_step = 0.005                      # time step thymio
max_motor_speed = 200                  # max motor speed thymio
GOAL = True                           # True if robot can see goal without having puck
evaluations = 275 # evaluation length (set to 300??)


# Set parameters social
popsize = 24


# Adjust pop size based on number of robots
with open('bots.txt', 'r') as f:
    data = f.read()
    data = data.splitlines()
    od_neat_size = 0
    for line in data:
        if len(line) > 3:
            od_neat_size += 1
    if '-o' not in sys.argv:
        od_neat_size = 1




EXPERIMENT_NAME = 'Foraging_ExperimentID' + str(od_neat_size) + '_' + ('G' if GOAL else 'NG') + "_" + str(sys.argv[1])

print 'STARTING', EXPERIMENT_NAME
print 'WITH POPSIZE', popsize

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
MAIN_LOG_PATH = os.path.join(CURRENT_FILE_PATH, 'log_main')
OUTPUT_PATH = os.path.join(CURRENT_FILE_PATH, 'output')
PICKLED_DIR = os.path.join(CURRENT_FILE_PATH, 'pickled')
FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')


class Foraging(TaskEvaluator):

    def __init__(self, commit_sha, debug=False, experimentName=EXPERIMENT_NAME):
        TaskEvaluator.__init__(self, debug, experimentName, evaluations=evaluations)
        self.ctrl_thread_started = False

        self.counter = 0
        self.camValues = np.array([0, 0 ,0 ,0 ,0 ,0]) # [puck_center[0], puck_center[1], puck_size, goal_center[0], goal_center[1], goal_size]
        self.timer = 0
        
        # Fitness of previous time steps is used to calculate fitness
        self.previous_fitnesses = 3*[0]
        self.haspuckhelper = False
    
        #set task params
        self.inputs=6                               # bias, see puck, has puck, see goal, front bumper, back bumper (all booleans)
        self.outputs=2                              # number of output nodes, can be one (turn/straight), or 2 (wheel speeds). the 2 is stil a TODO
        self.bias_as_node=False                     # set to false if you already put it in the inputs. Always set to false!
        self.weight_range=(-3., 3.)
        self.max_depth=4
        self.max_nodes=20
        self.response_default=1                     # bias in the node. what is this used for?
        self.feedforward=True
        
    
        # set social params
        self.sim_dis_neat = False                       # should be true when we use simulation
        self.phys_dis_neat= True                        # if we use simulation, the physical phys_dis_neat is false
        self.reset_innovations = False                  # reset innovation numbers after generations, false means the cNEAT algorithm
        self.generations = 20
        self.popsize = popsize / od_neat_size
    
        # NEAT population parameters
        self.elitism = True
        self.target_species = 2 #math.floor(popsize/2)  # target species is the pop size divided by 3
        self.tournament_selection_k= 2 # NOTE: was 1 in the original version. This is to increase selection pressure.
        self.compatibility_threshold=2.0
        self.compatibility_threshold_delta=0.1
        self.min_elitism_size=1 #???
        self.young_age=3
        self.young_multiplier=1.2 # for whole species
        self.stagnation_age=10        # number of generations the fitness does not increase befor it gets killed (not if having champ)
        self.old_age=7
        self.old_multiplier=0.5
        self.survival=0.6 # survival percentage of specie (NOTE: was 0.8 on working simple version)


        # NEAT genotype parameters
        self.types=['tanh']
        self.prob_add_node=0.05
        self.prob_add_conn=0.1
        self.prob_mutate = 0.4
        self.prob_mutate_weight=0.4 # prop to mutate weight for every connection
        self.prob_reset_weight=0.05
        self.prob_reenable_conn=0.01
        self.prob_disable_conn=0.01
        self.prob_reenable_parent=0.05  # chance to reenble connection
        self.prob_mutate_bias=0.2
        self.prob_mutate_response=0.0
        self.prob_mutate_type=0.2
        self.stdev_mutate_weight=1   # Used to init a newly added connection (original uses [-1,1] uniform) and to add to mutation
        self.stdev_mutate_bias=0.5
        self.stdev_mutate_response=0.5
        self.initial_weight_stdev=1.5
        self.distance_excess=1.0
        self.distance_disjoint=1.0
        self.distance_weight=0.7

    def evaluate(self, evaluee):
        if self.ctrl_client and not self.ctrl_thread_started:
            thread.start_new_thread(check_stop, (self, ))
            self.ctrl_thread_started = True
        return TaskEvaluator.evaluate(self, evaluee, self.counter)

    def _step(self, evaluee, callback):
        
        def ok_call(psValues):
            #obtain the camera information
            puck_center, puck_size = thymiotask.detector.detect_puck()
            goal_center, goal_size = thymiotask.detector.detect_goal()
            self.camValues = np.array([puck_center[0], puck_center[1], puck_size, goal_center[0], goal_center[1], goal_size])
            
            psValues = np.array([psValues[0], psValues[4], psValues[5], psValues[6]],dtype='f')
            
            # get sensor readings
            bumpingfront = 1 if (psValues[0]>1400 or psValues[1]>1400) else 0
            bumpingback = 1 if (psValues[2]>1500 or psValues[3]>1500) else 0

            

            
            hasPuck = 1 if puck_center[0] < thymiotask.detector.has_puck_threshold and puck_size > 0.05 else 0
            
            hasPuck = 1 if (hasPuck==1 or self.haspuckhelper==1) else 0
            
            # seeGoal = 1 if goal_size > 0.01 and goal_center[1] > 0.2 and goal_center[1] < 0.8 else 0
            seeGoal = 1 if goal_size > 0.01 else 0 # and goal_center[1] > 0.05 and goal_center[1] < 0.95 else 0
            seeGoal = 1 if (hasPuck or GOAL) and seeGoal else 0

            seePuck = 1 if (puck_size > 0.01 or hasPuck) else 0
            
            

            if seePuck and not hasPuck and not seeGoal:
                self.thymioController.SendEventName('SetColor', [0, 32, 0, 0], reply_handler=dbusReply, error_handler=dbusError)
            if hasPuck and not seeGoal:
                self.thymioController.SendEventName('SetColor', [0, 0, 32, 0], reply_handler=dbusReply, error_handler=dbusError)
            if seeGoal:
                self.thymioController.SendEventName('SetColor', [32, 0, 0, 0], reply_handler=dbusReply, error_handler=dbusError)
            if bumpingfront==1 or bumpingback==1:
                self.thymioController.SendEventName('SetColor', [32, 32, 0, 0], reply_handler=dbusReply, error_handler=dbusError)
            
                                
            

            # feed them in network and obtain motorspeed outputs
            #obsValues = np.concatenate([psValues, self.camValues])
            obsValues = np.asarray([puck_center[0], puck_center[1], puck_size, goal_center[0], goal_center[1], goal_size])
            inputValues = np.asarray([seePuck, hasPuck, seeGoal])

            
            # Feed with bias, bias is added as the first node input
            inputValues=np.hstack((1.0,inputValues,bumpingfront,bumpingback))

            # check if this is correct
            
            
            
            if self.outputs == 1:
                output = evaluee.feed(inputValues, add_bias=False)[-1:]

                if output < 0:
                    left = 1
                    right = 1
                else:
                    left = -0.7
                    right = 0.7
                motorspeed = { 'left': left, 'right': right }

            if self.outputs == 2:
                left, right = evaluee.feed(inputValues, add_bias=False)[-2:]
                motorspeed = { 'left': left, 'right': right }
            


            # send motorspeeds to the robot
            try:
                writeMotorSpeed(self.thymioController, motorspeed, max_motor_speed)
            except Exception as e:
                print str(e)

            callback(self.getFitness(motorspeed, obsValues, inputValues))

        def nok_call():
            print " Error while reading proximity sensors"

        getProxReadings(self.thymioController, ok_call, nok_call)
        return True



    def getFitness(self, motorspeed, observation, inputvals):
        if (inputvals[2]==1 and inputvals[3]==1 and self.timer <= 0 and self.counter > 3):
            print('GOAL!')
            fitness = 10000
            self.timer = 25
            self.counter = 0
            self.haspuckhelper = 0
        elif (inputvals[2]==1 and (sum(self.thymioController.GetVariable("thymio-II", "prox.ground.reflected")) < 1000) and (min(self.thymioController.GetVariable("thymio-II", "prox.ground.reflected")) < 380)):
            self.counter += 1
            fitness = 1000
        elif inputvals[1] == 1 and inputvals[2] == 1 and self.timer <= 0:
            self.haspuckhelper = 1
            fitness = 1000
            self.counter = 0
        else:
            fitness = 0
            self.counter = 0
        

        total_fitness = max(fitness - sum(self.previous_fitnesses), 0)
        
        if inputvals[4]==0 and inputvals[5]==0:
            total_fitness += 1
        
        self.previous_fitnesses.append(fitness)
        self.previous_fitnesses = self.previous_fitnesses[1:]
        if self.timer > 5:
            self.thymioController.SendEventName('SetColor', [1, 1, 1, 1], reply_handler=dbusReply, error_handler=dbusError)
            self.thymioController.SendEventName('PlayFreq', [700, 0], reply_handler=dbusReply, error_handler=dbusError)
            #left = -0.3
            #right = -0.3
            #motorspeed = { 'left': left, 'right': right }
            #writeMotorSpeed(self.thymioController, motorspeed, max_motor_speed)
            # ik kan hier nog implementeren dat de robot stil blijft staan voor altijd als de controller een expert controller is...
        else:
            self.thymioController.SendEventName('SetColor', [0, 0, 0, 0], reply_handler=dbusReply, error_handler=dbusError)
            self.thymioController.SendEventName('PlayFreq', [0, -1], reply_handler=dbusReply, error_handler=dbusError)

        self.timer -= 1
        return total_fitness



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
    ctrl_ip = sys.argv[-2]
    debug = False
    commit_sha = sys.argv[-1]
    foraging_task = Foraging(commit_sha, debug, EXPERIMENT_NAME)

    thymiotask = thymio_task.ThymioTask(foraging_task, OUTPUT_PATH=OUTPUT_PATH)

    thymiotask.start(foraging_task, foraging_task.popsize, foraging_task.generations, max_motor_speed, foraging=True)
