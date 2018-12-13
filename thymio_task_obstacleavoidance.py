# -*- coding: utf-8 -*-
from parameters import *
# from neat_task import NEATTask

import numpy as np
import dbus
import dbus.mainloop.glib
import logging
logging.basicConfig()
import parameters as pr
#import object_detector
#import cameracontroller
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
evaluations = 3000              # evaluation length


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





EXPERIMENT_NAME = 'Obstacleavoidance_ExperimentID' + str(od_neat_size) + '_'  + str(sys.argv[1])

print 'STARTING', EXPERIMENT_NAME
print 'WITH POPSIZE', popsize

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
MAIN_LOG_PATH = os.path.join(CURRENT_FILE_PATH, 'log_main')
OUTPUT_PATH = os.path.join(CURRENT_FILE_PATH, 'output')
PICKLED_DIR = os.path.join(CURRENT_FILE_PATH, 'pickled')
FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')


class Obstacleavoidance(TaskEvaluator):

    def __init__(self, commit_sha, debug=False, experimentName=EXPERIMENT_NAME):
        TaskEvaluator.__init__(self, debug, experimentName, evaluations=evaluations)
        self.ctrl_thread_started = False

    
        #set task params
        self.inputs=8                               # number of input nodes with bias node
        self.sensor_max= [4500,4500,4500,4500,4500,4500,4500] #max values for sensors
        self.sensor_max_act= [0,0,0,0,0,0,0] #activation due to obstacles
        
        self.outputs=2                              # wheel speeds
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
        self.tournament_selection_k=2
        self.compatibility_threshold=2.0
        self.compatibility_threshold_delta=0.1
        self.min_elitism_size=1 #???
        self.young_age=3
        self.young_multiplier=1.2 # for whole species
        self.stagnation_age=10        # number of generations the fitness does not increase befor it gets killed (not if having champ)
        self.old_age=7
        self.old_multiplier=0.5
        self.survival=0.6 # survival percentage of specie


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
            # get sensor readings
            psValues = np.array([psValues[0], psValues[1],psValues[2], psValues[3],psValues[4], psValues[5], psValues[6]],dtype='f')
            #print "psValues"
            #print psValues
            
            for i in range(self.inputs-1):
                #print "activated over threshold"
                psValues[i] = float(psValues[i])-float(self.sensor_max_act[i])
                #print psValues[i]
                #print "minus half of  range"
                psValues[i] =psValues[i] -  float((self.sensor_max[i]-self.sensor_max_act[i])/2.0)
                #print psValues[i]
                #print "divided half of  range"
                psValues[i] =psValues[i] / float((self.sensor_max[i]-self.sensor_max_act[i])/2.0)
                #print psValues[i]
                psValues[i] = min(1.0,psValues[i])
                psValues[i] = max(psValues[i],-1.0)
            
            #print "input"
            #print psValues
            
            # add bias node, note: there is a problem when the input is negative.
            psValues = np.hstack((1.0, psValues))
            
            #print "network"
            #print evaluee.cm
            #print "input"
            #print psValues
            #print "output"
        
            
            output = evaluee.feed(psValues, add_bias=False)
                                  
            #print output
                                  
            left, right = output[-2:]
            
            motorspeed = { 'left': left, 'right': right }
            
            try:
                writeMotorSpeed(self.thymioController, motorspeed)
            except Exception as e:
                print str(e)
            
            callback(self.getFitness(motorspeed, psValues))

        def nok_call():
            print " Error while reading proximity sensors"

        getProxReadings(self.thymioController, ok_call, nok_call)
        return True



    def getFitness(self, motorspeed, observation):

        speedpenalty = float(abs(motorspeed['left'] - motorspeed['right']))
        
        # Calculate normalized distance to the nearest object
        sensorpenalty = 0
        psValues = observation[-self.inputs+1:]
        for i in range(self.inputs-1):
            distance = psValues[i]
            if sensorpenalty < distance:
                sensorpenalty = distance



        # fitness for 1 timestep in [-2, 2], get it to [0,1]

        return float((motorspeed['left'] + motorspeed['right']+2.0)/4.0) * (1 - min(speedpenalty,1)) * (1 - min(sensorpenalty,1))



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
    debug = True
    commit_sha = sys.argv[-1]
    obstacleavoidance_task = Obstacleavoidance(commit_sha, debug, EXPERIMENT_NAME)

    thymiotask = thymio_task.ThymioTask(obstacleavoidance_task, OUTPUT_PATH=OUTPUT_PATH)

    thymiotask.start(obstacleavoidance_task, obstacleavoidance_task.popsize, obstacleavoidance_task.generations, max_motor_speed, foraging=False)
