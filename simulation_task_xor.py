# Created by Jesper Slik and modified by Jacqueline Heinerman in 2018
# Questions: j.v.heinerman@vu.nl
# This files sets up xor experiments. The evaluation of the Xor task is done in simulation_task_xor_evaluate

#!/usr/bin/env python
import os, time, sys, math
import base64
import pickle
import json
import numpy as np

from simulation_task_xor_evaluate import XORTask
from peas.methods.neat import NEATPopulation, NEATGenotype
from copy import deepcopy

# Set parameters: XOR task related and network
inputs=3                               # number of input nodes with bias node
outputs=2                              # number of output nodes
bias_as_node=False                      # set to false if you already put it in the inputs. Always set to false!
weight_range=(-3., 3.)
max_depth=5
max_nodes=10
response_default=1                      # bias in the nod. what is this used for?
feedforward=True

# Set parameters social
nBots = 3                              # number of bots, used as a proxy for number of robots (distributed system). when above 1, set dis_neat to True
sim_dis_neat = True                       # should be true when we use simulation
phys_dis_neat=False                        # if we use simulation, the physical phys_dis_neat is false
reset_innovations = False               # reset innovation numbers after generations, false means the cNEAT algorithm
nRun = 1                               # number of experiments (not used in physical tasks)

# NEAT population parameters
nGen = 20                     # number of generations
popSize = 8                            # population size per bot
elitism = True
target_species = math.floor(popSize/5)  # target species is the pop size divided by 3
tournament_selection_k=1
compatibility_threshold=1.0
compatibility_threshold_delta=0.4
min_elitism_size=3 #???
young_age=10
young_multiplier=1.2 # for whole species
stagnation_age=10        # number of generations the fitness does not increase befor it gets killed (not if having champ)
old_age=30
old_multiplier=0.2
survival=0.5 # survival percentage of specie


# NEAT genotype parameters
types=['tanh']
prob_add_node=0.1
prob_add_conn=0.3
prob_mutate = 0.25
prob_mutate_weight=0.4 # prop to mutate weight for every connection
prob_reset_weight=0.1
prob_reenable_conn=0.01
prob_disable_conn=0.01
prob_reenable_parent=0.1  # chance to reenble connection
prob_mutate_bias=0.2
prob_mutate_response=0.0
prob_mutate_type=0.2
stdev_mutate_weight=1   # Used to init a newly added connection (original uses [-1,1] uniform) and to add to mutation
stdev_mutate_bias=0.5
stdev_mutate_response=0.5
initial_weight_stdev=2.0
distance_excess=1.0
distance_disjoint=1.0
distance_weight=0.4



# log location of bots as a proxy for the ip adresses of the robots.
sim_geno_loc = os.getcwd() + "/output/xor/genos" + time.strftime("%Y%m%d-%H%M%S", time.gmtime())
sim_geno_loc_out = sim_geno_loc + 'output'
if sim_dis_neat and not os.path.isdir(sim_geno_loc):
    os.makedirs(sim_geno_loc)
    os.makedirs(sim_geno_loc_out)



def log_parameters(log): #TODO ADD ALL PARAMS

    log['parameters'] = {
    'inputs': inputs,
    'outputs': outputs,
    'bias_as_node' : bias_as_node ,
    'max_depth':  max_depth,
    'max_nodes' :  max_nodes,
    'response_default' :  response_default,
    'feedforward' :  feedforward,
    'reset_innovations' :  reset_innovations,
    'elitism' : elitism ,
    'target_species' : target_species ,
    'tournament_selection_k' :tournament_selection_k  ,
    'compatibility_threshold' :compatibility_threshold  ,
    'compatibility_threshold_delta' : compatibility_threshold_delta ,
    'min_elitism_size' : min_elitism_size ,
    'young_age' :  young_age,
    'young_multiplier':young_multiplier,
    'stagnation_age':stagnation_age,
    'old_age': old_age,
    'old_multiplier': old_multiplier,
    'survival': survival,
    'types': types,
    'prob_add_node': prob_add_node,
    'prob_add_conn': prob_add_conn,
    'prob_mutate': prob_mutate,
    'prob_mutate_weight': prob_mutate_weight,
    'prob_reenable_conn': prob_reenable_conn,
    'prob_disable_conn': prob_disable_conn,
    'prob_reenable_parent': prob_reenable_parent,
    'prob_mutate_bias': prob_mutate_bias,
    'prob_mutate_response': prob_mutate_response,
    'prob_mutate_type': prob_mutate_type,
    'stdev_mutate_weight': stdev_mutate_weight,
    'stdev_mutate_bias': stdev_mutate_bias,
    'stdev_mutate_response' : stdev_mutate_response,
    'initial_weight_stdev' : initial_weight_stdev,
    'distance_excess' : distance_excess,
    'distance_disjoint': distance_disjoint,
    'distance_weight' :distance_weight
        
        }
    return log


def log_population(log,run, botnumber, pop):
    pop_backup = pop.giveBackUp()
    species_backup = pop.giveBackUpSpecies()
    generation = {'robot': botnumber, 'gen_number': pop.generation , 'individuals': [] }
    for individual in pop_backup:
        copied_connections = {str(key): value for key, value in individual.conn_genes.items()}
        generation['individuals'].append({
                                'stats': deepcopy(individual.stats),
                                'node_genes': deepcopy(individual.node_genes),
                                'conn_genes': copied_connections,
                                })
    generation['species_id'] = [species.id for species in species_backup]
    generation['species_size'] = [len(species.members) for species in species_backup]
    log['generations'].append(generation)

    filename = sim_geno_loc_out + '/run_' + str(run)  + '/generation_' + str(pop.generation) + '.txt'
    with open(filename, 'w') as f:
        json.dump(log, f)


            
def log_summary(run, botnumber, pop):
    fitness = []
    nodes = []
    conns = []
    pop_backup = pop.giveBackUp()
    species_backup = pop.giveBackUpSpecies()
    
    for specie in species_backup:
        print "species id"
        print specie.id
        print "species size"
        print len(specie.members)

    for i in pop_backup:
        fitness.append(i.stats['fitness'])
        nodes.append(len(i.node_genes))
        conns.append(len(i.conn_genes))
        
    g = pop.generation
    f1 = min(fitness)
    f2 = np.percentile(fitness, 25)
    f3 = np.percentile(fitness, 50)
    f4 = np.percentile(fitness, 75)
    f5 = max(fitness)
    n1 = min(nodes)
    n2 = np.percentile(nodes, 25)
    n3 = np.percentile(nodes, 50)
    n4 = np.percentile(nodes, 75)
    n5 = max(nodes)
    c1 = min(conns)
    c2 = np.percentile(conns, 25)
    c3 = np.percentile(conns, 50)
    c4 = np.percentile(conns, 75)
    c5 = max(conns)
    msg= "%d\t%d\t%f\t%f\t%f\t%f\t%f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d" % (botnumber, g, f1, f2, f3, f4, f5, n1, n2, n3, n4, n5, c1, c2, c3, c4, c5)

    filename = sim_geno_loc_out + '/run_' + str(run)  + '/summary.txt'
    with open(filename, 'a') as f:
        f.write(msg + "\n")


def sim_run(popSize, nGen, nBots, nRun, sim_geno_loc, sim_dis_neat, reset_innovations):
    pops = []
    for robot in xrange(nBots):
        # create a different population for every bot
        pops.append(NEATPopulation(genotype,
                                   popsize=popSize,
                                   elitism=elitism,
                                   compatibility_threshold=compatibility_threshold,
                                   compatibility_threshold_delta=compatibility_threshold_delta,
                                   target_species=target_species,
                                   min_elitism_size=min_elitism_size,
                                   young_age=young_age,
                                   prob_mutate = prob_mutate,
                                   young_multiplier=young_multiplier,
                                   stagnation_age=stagnation_age,
                                   old_age=old_age,
                                   old_multiplier=old_multiplier,
                                   tournament_selection_k=tournament_selection_k,
                                   reset_innovations=reset_innovations,
                                   survival=survival,
                                   phys_dis_neat=phys_dis_neat,
                                   sim_dis_neat=sim_dis_neat,
                                   sim_geno_loc=sim_geno_loc + '/' + str(robot) + '.pkl'
                                   ))
    
    log_dump = {'parameters': {}, 'generations': []}
    log_dump = log_parameters(log_dump)
    
    
    for i in xrange(nRun):
        print time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + "\t%d" %i
        # create summary file for indidivuals over all populations
        runfile = sim_geno_loc_out + '/run_' + str(i)
        os.makedirs(runfile)
        
        for gen in xrange(nGen):
            botnumber=1
            for pop in pops:
                pop.epoch(generations=1, evaluator=task, solution=task, reset=(gen==0))
                # log population dump per robot
                log_population(log_dump, i, botnumber, pop)
                log_summary(i,botnumber,pop)
                botnumber+=1






genotype = lambda innovations={}: NEATGenotype(inputs=inputs,
                                               outputs=outputs,
                                               weight_range=weight_range,
                                               types=types,
                                               innovations=innovations,
                                               feedforward=feedforward,
                                               prob_add_node=prob_add_node,
                                               prob_add_conn=prob_add_conn,
                                               prob_mutate_weight=prob_mutate_weight,
                                               prob_reset_weight=prob_reset_weight,
                                               prob_reenable_conn=prob_reenable_conn,
                                               prob_disable_conn=prob_disable_conn,
                                               prob_reenable_parent=prob_reenable_parent,
                                               prob_mutate_bias=prob_mutate_bias,
                                               prob_mutate_response=prob_mutate_response,
                                               prob_mutate_type=prob_mutate_type,
                                               stdev_mutate_weight=stdev_mutate_weight,
                                               stdev_mutate_bias=stdev_mutate_bias,
                                               stdev_mutate_response=stdev_mutate_response,
                                               phys_dis_neat=phys_dis_neat,
                                               max_depth=max_depth,
                                               max_nodes=max_nodes,
                                               response_default=response_default,
                                               initial_weight_stdev=initial_weight_stdev,
                                               bias_as_node=bias_as_node,
                                               distance_excess=distance_excess,
                                               distance_disjoint=distance_disjoint,
                                               distance_weight=distance_weight
                                               
                                               )
task = XORTask()



print 'Starting simulation: ' + str([popSize, nGen, nBots, nRun])

sim_run(popSize, nGen, nBots, nRun, sim_geno_loc, sim_dis_neat, reset_innovations=reset_innovations)
