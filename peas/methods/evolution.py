""" This module implements the different genotypes and evolutionary
    methods used in NEAT and HyperNEAT. It is meant to be a reference
    implementation, so any inneficiencies are copied as they are
    described.
"""
# this file runs the general evolutionary algorithm components. The actual implementation of the evolve step is done in neat.py

### IMPORTS ###
import sys
import random
import multiprocessing
from copy import deepcopy
from itertools import product
from collections import defaultdict
from ..networks.rnn import NeuralNetwork
import numpy as np
np.seterr(over='warn', divide='raise')
import logging
import os.path

# Shortcuts
rand = random.random
inf  = float('inf')

### FUNCTIONS ###

# This function is called in the evaluate_all further down.
def evaluate_individual((individual, evaluator, phenotype)):
    #print "evaluate_individual in evolution.py "
    try:
        NeuralNetwork(phenotype)
    except:
        ## Some weird situation that NEATGenotye cannot be converted to a nn
        ## Occurs nearly never, but once in ~100000 evals, for a presumably infeasible geno, and only whilst simulating dNEAT
        print 'Cannot convert NEATGenotype to nn! Assigning a fitness of 0.  evolution.py -> evaluate_individual()'
        individual.stats = {}
        individual.stats['id'] = -1
        individual.stats['parent1'] = -1
        individual.stats['parent2'] = -1
        individual.stats['fitness'] = 0
        return individual
            
            #print "go to task file"
    if callable(evaluator):
        individual.stats = evaluator(phenotype)
    elif hasattr(evaluator, 'evaluate'):
        individual.stats = evaluator.evaluate(phenotype)
    else:
        raise Exception("Evaluator must be a callable or object" \
                        "with a callable attribute 'evaluate'.")
    individual.stats['id'] = individual.id
    individual.stats['parent1'] = individual.parent1
    individual.stats['parent2'] = individual.parent2
    individual.stats['specieid'] = individual.specieid
    return individual


### CLASSES ###
                
class SimplePopulation(object):
    
    #OPEN QUESTION: seems that these parameters are used. Not the point
    def __init__(self, geno_factory,
                       stop_when_solved=False,
                       verbose=True,
                       max_cores=1):
        # Instance properties
        self.geno_factory           = geno_factory
        self.stop_when_solved       = stop_when_solved
        self.verbose                = verbose
        self.max_cores              = max_cores
        


        cpus = multiprocessing.cpu_count()
        use_cores = min(self.max_cores, cpus-1)
        if use_cores > 1:
            self.pool = multiprocessing.Pool(processes=use_cores, maxtasksperchild=5)
        else:
            self.pool = None
        
    def _reset(self):
        """ Resets the state of this population.
        """
        self.population      = [] # List of individuals
        self.phenotypes      = []
                
        # Keep track of some history
        self.champions  = []
        self.generation = 0
        self.solved_at  = None
        self.stats = defaultdict(list)
                
    def epoch(self, evaluator, generations, expertBoolean, solution=None, reset=True, callback=None, converter=lambda x: x):

        """ Runs an evolutionary epoch 

            :param evaluator:    Either a function or an object with a function
                                 named 'evaluate' that returns a given individual's
                                 fitness.
            :param callback:     Function that is called at the end of each generation.
            :param converter:    Function that converts the neat genotype in another network
        """
        self.reset = reset
        if self.reset:
            self._reset()
        
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('simulationLogger')
        
        
        if expertBoolean == True:
            while True:
                startBoolean = False
                startExpert = False
                if os.path.exists('./dummy.txt'):
                    startBoolean = True
                if os.path.exists('./expert.txt'):
                    startExpert = True
                self._evolve_expert(evaluator, converter, startBoolean, startExpert, solution)
    
        if expertBoolean == False:
            for _ in xrange(generations):
                logger.info('Generation: %d' % _)
                self._evolve(evaluator, converter, solution)
                self.generation += 1
                callback(self)
#if callback is not None:
#           callback(self)


                #self.generation += 1
            
            #if callback is not None:
            #   callback(self)
                
                #if self.solved_at is not None and self.stop_when_solved:
                #break
                
        return {'stats': self.stats, 'champions': self.champions}
    

        
    def _evaluate_all(self, pop, evaluator, converter):
        """ Evaluates all of the individuals in given pop,
            and assigns their "stats" property.
        """
        #print "evaluate all individuals"
        to_eval = [(individual, evaluator, converter(individual)) for individual in pop]
        self.phenotypes = [i[2] for i in to_eval]
        if self.pool is not None:
            pop = self.pool.map(evaluate_individual, to_eval)
        else:
            pop = map(evaluate_individual, to_eval)
        
        return pop
    
    def _find_best(self, pop, converter, solution=None):
        """ Finds the best individual of the population, and adds it to the champions list, also
            checks if this best individual 'solves' the problem.
        """
        ## CHAMPION
        self.champions.append(max(pop, key=lambda ind: ind.stats['fitness']))
        
        ## SOLUTION CRITERION
        if solution is not None:
            champion = converter(self.champions[-1])
            if isinstance(solution, (int, float)):
                solved = (champion.stats['fitness'] >= solution)
            elif callable(solution):
                solved = solution(champion)
            elif hasattr(solution, 'solve'):
                solved = solution.solve(champion)
            else:
                raise Exception("Solution checker must be a threshold fitness value,"\
                                "a callable, or an object with a method 'solve'.")
            
            if solved and self.solved_at is None:
                self.solved_at = self.generation
    


        
