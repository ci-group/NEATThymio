# Modified by Jacqueline Heinerman in 2018
# Questions: j.v.heinerman@vu.nl
# This class evaluates 1 individual with a fitness max(fitness, 1). Make sure the physical task work with this minimum (this minimum is neccesary with fitness proportional selection, minimum then can't be zero)
# This class does not need to be changed when creating a new task.

import os
import time
import gobject
import glib
import dbus
import dbus.mainloop.glib
import logging
import thread
from helpers import *
from parameters import *
import peas.methods.hyperneat as hn

class TaskEvaluator:
    #these parameters are set in the specific physical task
    def __init__(self, debug=False, experimentName='task', evaluations=200, timeStep=0.005, activationFunction='tanh', popSize=1, generations=100, solvedAt=1000):

        self.logger = logging.getLogger('simulationLogger')
        logLevel = logging.INFO
        if debug:
            logLevel = logging.DEBUG
        self.logger.setLevel(logLevel)
        self.experimentName = experimentName

        self.counter = 0


        self.evaluations = evaluations
        self.timeStep = timeStep
        self.activationFunction = activationFunction
        self.popSize = popSize
        self.generations = generations
        self.solvedAt = solvedAt
        self.ctrl_client = None
        self.thymioController = None
        self._use_hyperneat = False


    # Override this to set controller parameters on initialization
    def set_thymio_controller(self, controller):
        self.thymioController = controller

    def set_ctrl_client(self, ctrl_client):
        self.ctrl_client = ctrl_client

    def _step(self, evaluee, callback):
        raise NotImplemented('Step method not implemented')

    def evaluate(self, evaluee, counter):
        self.evaluations_taken = 0
        self.fitness = 0
        self.loop = gobject.MainLoop()
        def update_fitness(task, fit):
            task.fitness += fit
        def main_lambda(task, counter):
            if task.evaluations_taken >= self.evaluations:
                stopThymio(self.thymioController)
                task.loop.quit()
                return False
            # now the _step function of the specific task is called
            ret_value =  self._step(evaluee, lambda (fit): update_fitness(self, fit))
            task.evaluations_taken += 1
            return ret_value
        gobject.timeout_add(int(self.timeStep * 1000), lambda: main_lambda(self, counter))
        # glib.idle_add(lambda: main_lambda(self))
        self.loop.run()

        fitness = max(self.fitness, 1)

        self.logger.info('Fitness at end in task eval: %d' % fitness)

        time.sleep(1)

        return { 'fitness': fitness }

    def solve(self, evaluee):
        return int(self.evaluate(evaluee)['fitness'] >= self.solvedAt)

    def getFitness(self, motorspeed, observation):
        raise NotImplemented('Fitness method not implemented')

    def getLogger(self):
        return self.logger

    def exit(self, value = 0):
        print 'Exiting...'
        # sys.exit(value)
        self.loop.quit()
        # cleanup_stop_thread()
        thread.interrupt_main()

    def logs(self):
        return {}

