""" This module implements the different genotypes and evolutionary
    methods used in NEAT and HyperNEAT. It is meant to be a reference
    implementation, so any inneficiencies are copied as they are
    described.
"""

### IMPORTS ###
import sys
import random
import time
from copy import deepcopy
from itertools import product
from collections import defaultdict
import socket

from MessageReceiver import *
from MessageSender import *
from Inbox import Inbox
from ConnectionsListener import ConnectionsListener

import cPickle as pickle

# Libs
import numpy as np
np.seterr(over='warn', divide='raise')

# Package
from .evolution import SimplePopulation
from ..networks.rnn import NeuralNetwork

# Shortcuts
rand = random.random
inf  = float('inf')

# calculate microseconds
current_micro_time = lambda: int(round(time.time() * 1000000))

# calculate innovation number
innovation_number = lambda(inv): float(str(inv) + '.' + str(current_micro_time()))

### CLASSES ###
                
class NEATGenotype(object):
    """ Implements the NEAT genotype, consisting of
        node genes and connection genes.
    """
    def __init__(self, 
                 inputs=2, 
                 outputs=1, 
                 types=['tanh'],
                 topology=None,
                 feedforward=True,
                 max_depth=None,
                 max_nodes=inf,
                 response_default=4.924273,
                 initial_weight_stdev=2.0,
                 bias_as_node=False,
                 prob_add_node=0.03,
                 prob_add_conn=0.3,
                 prob_mutate_weight=0.8,
                 prob_reset_weight=0.1,
                 prob_reenable_conn=0.01,
                 prob_disable_conn=0.01,
                 prob_reenable_parent=0.25,
                 prob_mutate_bias=0.2,
                 prob_mutate_response=0.0,
                 prob_mutate_type=0.2,
                 stdev_mutate_weight=1.5,
                 stdev_mutate_bias=0.5,
                 stdev_mutate_response=0.5,
                 weight_range=(-50., 50.),
                 distance_excess=1.0,
                 distance_disjoint=1.0,
                 distance_weight=0.4):
        """ Refer to the NEAT paper for an explanation of these parameters.
        """

        # Settings
        self.inputs = inputs
        self.outputs = outputs
        
        # Restrictions
        self.types = types
        self.feedforward = feedforward
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        
        # Settings
        self.response_default = response_default
        self.initial_weight_stdev = initial_weight_stdev
        self.bias_as_node = bias_as_node
        
        # Mutation probabilities
        self.prob_add_conn = prob_add_conn
        self.prob_add_node = prob_add_node
        self.prob_mutate_weight = prob_mutate_weight
        self.prob_reset_weight = prob_reset_weight      
        self.prob_reenable_conn = prob_reenable_conn
        self.prob_disable_conn = prob_disable_conn
        self.prob_reenable_parent = prob_reenable_parent
        self.prob_mutate_bias = prob_mutate_bias
        self.prob_mutate_response = prob_mutate_response
        self.prob_mutate_type = prob_mutate_type
        
        # Mutation effects
        self.stdev_mutate_weight = stdev_mutate_weight
        self.stdev_mutate_bias = stdev_mutate_bias
        self.stdev_mutate_response = stdev_mutate_response
        self.weight_range = weight_range
        
        # Species distance measure
        self.distance_excess = distance_excess
        self.distance_disjoint = distance_disjoint
        self.distance_weight = distance_weight
        
        self.node_genes = []  #: Tuples of (fforder, type, bias, response, layer)
        #: Tuples of (innov, from, to, weight, enabled), indexed by input and output nodes.
        # For odneat, the innovation number is initially added as chronological integer, but new connections will
        # be appended with a timestamp to ensure chronological ordering with external genotypes
        self.conn_genes = {}
        
        if self.bias_as_node:
            self.inputs += 1
            
        max_layer = sys.maxint if (self.max_depth is None) else (self.max_depth - 1)
        
        if topology is None:
            # The default method of creating a genotype
            # is to create a so called "fully connected"
            # genotype, i.e., connect all input nodes
            # to the output node.
            
            # Create input nodes
            for i in xrange(self.inputs):
                # We set the 'response' to 4.924273. Stanley doesn't mention having the response
                # be subject to evolution, so this is #weird, but we'll do it because neat-python does.
                self.node_genes.append( [i * 1024.0, types[0], 0.0, self.response_default, 0] )
            
            # Create output nodes
            for i in xrange(self.outputs):
                self.node_genes.append( [(self.inputs + i) * 1024.0, random.choice(self.types), 
                                            0.0, self.response_default, max_layer] )
            
            # Create connections from each input to each output
            innov = 0
            for i in xrange(self.inputs):
                for j in xrange(self.inputs, self.inputs + self.outputs):
                    self.conn_genes[(i, j)] = [innov, i, j, np.random.normal(0.0, self.initial_weight_stdev), True]
                    innov += 1
        else:
            # If an initial topology is given, use that:
            fr, to = zip(*topology)
            maxnode = max(max(fr), max(to))

            if maxnode + 1 < inputs + outputs:
                raise Exception("Topology (%d) contains fewer than inputs (%d) + outputs (%d) nodes." % 
                    (maxnode, inputs, outputs))
                
            for i in xrange(maxnode+1):
                # Assign layer 0 to input nodes, otherwise just an incrementing number,
                # i.e. each node is on its own layer.
                layer = 0 if i < inputs else i + 1
                self.node_genes.append( [i * 1024.0, random.choice(self.types), 0.0, self.response_default, layer] )
            innov = 0
            for fr, to in topology:
                self.conn_genes[(fr, to)] = [innov, fr, to, np.random.normal(0.0, self.initial_weight_stdev), True]
                innov += 1

    def mutate(self, innovations={}, global_innov=0):
        """ Perform a mutation operation on this genotype. 
            If a dict with innovations is passed in, any
            add connection operations will be added to it,
            and checked to ensure identical innovation numbers.
        """

        #  retrieve the maximum innovation number and cast this to an integer, so timestamp information is not taken
        maxinnov = int(max(global_innov, max(int(cg[0]) for cg in self.conn_genes.values())))
        
        if len(self.node_genes) < self.max_nodes and rand() < self.prob_add_node:
            # retrieve all the keys of the connections, which are tupels of nodes.
            possible_to_split = self.conn_genes.keys()
            # If there is a max depth, we can only split connections that skip a layer.
            # E.g. we can split a connection from layer 0 to layer 2, because the resulting
            # node would be in layer 1. We cannot split a connection from layer 1 to layer 2,
            # because that would create an intra-layer connection.
            if self.max_depth is not None:
                possible_to_split = [(fr, to) for (fr, to) in possible_to_split if
                                        self.node_genes[fr][4] + 1 < self.node_genes[to][4]]
            if possible_to_split:
                to_split = self.conn_genes[random.choice(possible_to_split)]
                to_split[4] = False  # Disable the old connection
                fr, to, w = to_split[1:4]
                avg_fforder = (self.node_genes[fr][0] + self.node_genes[to][0]) * 0.5
                # We assign a random function type to the node, which is #weird
                # because I thought that in NEAT these kind of mutations
                # initially don't affect the functionality of the network.
                new_type = random.choice(self.types)
                # We assign a 'layer' to the new node that is one lower than the target of the connection
                layer = self.node_genes[fr][4] + 1
                node_gene = [avg_fforder, new_type, 0.0, self.response_default, layer]
                # new_id = innovation_number(len(self.node_genes)-1)
                self.node_genes.append(node_gene)
                # give the newly created node a label for the connection
                new_id = len(self.node_genes) - 1

                if (fr, new_id) in innovations:
                    innov = innovations[(fr, new_id)]
                else:
                    # maxinnov += 1
                    # assign a new innovation number based on last innovation number with a timestamp.
                    innov = innovations[(fr, new_id)] = innovation_number(maxinnov)
                self.conn_genes[(fr, new_id)] = [innov, fr, new_id, 1.0, True]

                if (new_id, to) in innovations:
                    innov = innovations[(new_id, to)]
                else:
                    # maxinnov += 1
                    innov = innovations[(new_id, to)] = innovation_number(maxinnov)
                self.conn_genes[(new_id, to)] = [innov, new_id, to, w, True]
            
        # This is #weird, why use "elif"? but this is what
        # neat-python does, so I'm copying.
        elif rand() < self.prob_add_conn:
            potential_conns = product(xrange(len(self.node_genes)), xrange(self.inputs, len(self.node_genes)))
            potential_conns = (c for c in potential_conns if c not in self.conn_genes)
            # Filter further connections if we're looking only for FF networks
            if self.feedforward:
                potential_conns = ((f, t) for (f, t) in potential_conns
                                   if self.node_genes[f][0] < self.node_genes[t][0])  # Check FFOrder
            # Don't create intra-layer connections if there is a max_depth
            if self.max_depth is not None:
                potential_conns = ((f, t) for (f, t) in potential_conns if
                                   self.node_genes[f][4] < self.node_genes[t][4]) # Check Layer
            potential_conns = list(potential_conns)
            # If any potential connections are left
            if potential_conns:
                (fr, to) = random.choice(potential_conns)
                # Check if this innovation was already made, otherwise assign max + 1
                if (fr, to) in innovations:
                    innov = innovations[(fr, to)]
                else:
                    # maxinnov += 1
                    innov = innovations[(fr, to)] = innovation_number(maxinnov)
                conn_gene = [innov, fr, to, np.random.normal(0, self.stdev_mutate_weight), True]
                self.conn_genes[(fr, to)] = conn_gene
            
        else:
            for cg in self.conn_genes.values():
                if rand() < self.prob_mutate_weight:
                    cg[3] += np.random.normal(0, self.stdev_mutate_weight)
                    cg[3] = np.clip(cg[3], self.weight_range[0], self.weight_range[1])

                if rand() < self.prob_reset_weight:
                    cg[3] = np.random.normal(0, self.stdev_mutate_weight)
                    
                if rand() < self.prob_reenable_conn:
                    cg[4] = True

                if rand() < self.prob_disable_conn:
                    cg[4] = False
                    
            # Mutate non-input nodes
            for node_gene in self.node_genes[self.inputs:]:
                if rand() < self.prob_mutate_bias:
                    node_gene[2] += np.random.normal(0, self.stdev_mutate_bias)
                    node_gene[2] = np.clip(node_gene[2], self.weight_range[0], self.weight_range[1])
                    
                if rand() < self.prob_mutate_type:
                    node_gene[1] = random.choice(self.types)
                    
                if rand() < self.prob_mutate_response:
                    node_gene[3] += np.random.normal(0, self.stdev_mutate_response)
                    
        for (fr, to) in self.conn_genes:
            if self.node_genes[int(to)][4] == 0:
                raise Exception("Connection TO input node not allowed.")
        return self  # For chaining
        
    def mate(self, other):
        """ Performs crossover between this genotype and another,
            and returns the child
        """
        child = deepcopy(self)
        child.node_genes = []
        child.conn_genes = {}
        # Select node genes from parents
        maxnodes = max(len(self.node_genes), len(other.node_genes))
        minnodes = min(len(self.node_genes), len(other.node_genes))
        for i in range(maxnodes):
            ng = None
            if i < minnodes:
                ng = random.choice((self.node_genes[i], other.node_genes[i]))
            else:
                try:
                    ng = self.node_genes[i]
                except IndexError:
                    ng = other.node_genes[i]
            child.node_genes.append(deepcopy(ng))
                
        # index the connections by innov numbers
        # self_conns = dict( ((c[0], c) for c in self.conn_genes.values()) )
        # other_conns = dict( ((c[0], c) for c in other.conn_genes.values()) )
        # maxinnov = max( self_conns.keys() + other_conns.keys() )

        #index connections by from-to pairs
        self_conns2 = dict((((c[1], c[2]), c) for c in self.conn_genes.values()))
        other_conns2 = dict((((c[1], c[2]), c) for c in other.conn_genes.values()))

        for fromto, con in self_conns2.iteritems():
            if fromto in other_conns2.keys():
                # the connection also exists in the other genotype
                cg = random.choice((con, other_conns2[fromto]))
                enabled = con[4] and other_conns2[fromto][4]
            else:
                cg = con
                enabled = cg[4]
            if cg is not None:
                child.conn_genes[(cg[1], cg[2])] = deepcopy(cg)
                child.conn_genes[(cg[1], cg[2])][4] = enabled or rand() < self.prob_reenable_parent

        # for i in range(maxinnov+1):
        #     cg = None
        #     if i in self_conns and i in other_conns: # also check if from and to are the same.
        #         cg = random.choice((self_conns[i], other_conns[i]))
        #         enabled = self_conns[i][4] and other_conns[i][4]
        #     else:
        #         if i in self_conns:
        #             cg = self_conns[i]
        #             enabled = cg[4]
        #         elif i in other_conns:
        #             cg = other_conns[i]
        #             enabled = cg[4]
        #     if cg is not None:
        #         child.conn_genes[(cg[1], cg[2])] = deepcopy(cg)
        #         child.conn_genes[(cg[1], cg[2])][4] = enabled or rand() < self.prob_reenable_parent

        # Filter out connections that would become recursive in the new individual.

        class fr_to_count:
            fr_count = 1
            to_count = 1

        def is_feedforward(((fr, to), cg)):
            if isinstance(fr, float):
                fr = fr + fr_to_count.fr_count
                fr_to_count.fr_count += 1
            if isinstance(to, float):
                to = to + fr_to_count.to_count
                fr_to_count.to_count += 1
            return child.node_genes[fr][0] < child.node_genes[to][0]

        if self.feedforward:
            try:
                child.conn_genes = dict(filter(is_feedforward, child.conn_genes.items()))
            except:
                print "feedforward filter failed"

        return child
        
    def distance(self, other):
        """ NEAT's compatibility distance
        """
        # index the connections by innov numbers
        self_conns = dict( ((c[0], c) for c in self.conn_genes.itervalues()) )
        other_conns = dict( ((c[0], c) for c in other.conn_genes.itervalues()) )
        # Select connection genes from parents
        allinnovs = self_conns.keys() + other_conns.keys()
        mininnov = min(allinnovs)
        
        e = 0
        d = 0
        w = 0.0
        m = 0
        
        for i in allinnovs:
            if i in self_conns and i in other_conns:
                w += np.abs(self_conns[i][3] - other_conns[i][3])
                m += 1
            elif i in self_conns or i in other_conns:
                if i < mininnov:
                    d += 1  # Disjoint
                else:
                    e += 1  # Excess
            else:
                raise Exception("NEITHERHWATHWATHAWTHA")
        
        w = (w / m) if m > 0 else w
        
        return (self.distance_excess * e + 
                self.distance_disjoint * d +
                self.distance_weight * w)
                
    def get_network_data(self):
        """ Returns a tuple of (connection_matrix, node_types) 
            that is reordered by the "feed-forward order" of the network,
            Such that if feedforward was set to true, the matrix will be
            lower-triangular.
            The node bias is inserted as "node 0", the leftmost column
            of the matrix.
        """
        
        # Assemble connectivity matrix
        cm = np.zeros((len(self.node_genes), len(self.node_genes)))
        cm.fill(np.nan)
        count_fr = 1
        count_to = 1
        for (_, fr, to, weight, enabled) in [c[1] for c in sorted(self.conn_genes.iteritems(), key=lambda x: x[1])]:
            if enabled:
                # check if the nodes are appended with a timestamp. If so, convert to an index in de matrix
                if isinstance(fr, float):
                    fr = int(fr) + count_fr
                    count_fr += 1
                if isinstance(to, float):
                    to = int(to) + count_to
                    count_to += 1
                try:
                    cm[to, fr] = weight
                except IndexError:
                    pass
        
        # Reorder the nodes/connections
        ff, node_types, bias, response, layer = zip(*self.node_genes)
        order = [i for _,i in sorted(zip(ff, xrange(len(ff))))]
        cm = cm[:,order][order,:]
        node_types = np.array(node_types)[order]
        bias = np.array(bias)[order]
        response = np.array(response)[order]
        layers = np.array(layer)[order]

        # Then, we multiply all the incoming connection weights by the response
        cm *= np.atleast_2d(response).T
        # Finally, add the bias as incoming weights from node-0
        if not self.bias_as_node:
            cm = np.hstack( (np.atleast_2d(bias).T, cm) )
            cm = np.insert(cm, 0, 0.0, axis=0)
            # TODO: this is a bit ugly, we duplicate the first node type for 
            # bias node. It shouldn't matter though since the bias is used as an input.
            node_types = [node_types[0]] + list(node_types)

        if self.feedforward and np.triu(np.nan_to_num(cm)).any():
            import pprint
            pprint.pprint(self.node_genes)
            pprint.pprint(self.conn_genes)
            print ff
            print order
            print np.sign(cm)
            raise Exception("is feedforward.")
        
        return cm, node_types
                
    def __str__(self):
        return '%s with %d nodes and %d connections.' % (self.__class__.__name__, 
            len(self.node_genes), len(self.conn_genes))
            
    def visualize(self, filename):
        return NeuralNetwork(self).visualize(filename, inputs=self.inputs, outputs=self.outputs)
        
class NEATSpecies(object):
    
    def __init__(self, initial_member):
        self.members = [initial_member]
        self.representative = initial_member
        self.offspring = 0
        self.age = 0
        self.avg_fitness = 0.
        self.max_fitness = 0.
        self.max_fitness_prev = 0.
        self.no_improvement_age = 0
        self.has_best = False
        
class NEATPopulation(SimplePopulation):
    """ A population object for NEAT, it contains the selection
        and reproduction methods.
    """
    def __init__(self, ip_address,
                 geno_factory,
                 compatibility_threshold=3.0,
                 compatibility_threshold_delta=0.4,
                 target_species=12,
                 min_elitism_size=5,
                 young_age=10,
                 young_multiplier=1.2,
                 stagnation_age=15,
                 old_age=30,
                 old_multiplier=0.2,
                 reset_innovations=False,
                 survival=0.2,
                 **kwargs):
        """ Initializes the object with settings,
            does not create a population yet.
            
            :param geno_factory: A callable (function or object) that returns
                                 a new instance of a genotype.

        """
        super(NEATPopulation, self).__init__(geno_factory, **kwargs)
        self.ip_address = ip_address
        self.compatibility_threshold = compatibility_threshold
        self.compatibility_threshold_delta = compatibility_threshold_delta
        self.target_species = target_species
        self.reset_innovations = reset_innovations
        self.survival = survival
        self.young_age = young_age
        self.young_multiplier = young_multiplier
        self.old_age = old_age
        self.old_multiplier = old_multiplier
        self.stagnation_age = stagnation_age
        self.min_elitism_size = min_elitism_size
        self.logger = Logger()
        self.distributor = Distributor(self.ip_address, self.logger)
        self.distributor.start()
        self.population_backup = self.population
        self.species_backup = []


    def _reset(self):
        """ Resets the state of this population.
        """
        # Keep track of some history
        self.champions = []
        self.generation = 0
        self.solved_at = None
        self.stats = defaultdict(list)
        
        # Neat specific:
        self.species = [] # List of species
        self.global_innov = 0
        self.innovations = {} # Keep track of global innovations
        self.current_compatibility_threshold = self.compatibility_threshold
                
    @property
    def population(self):
        for specie in self.species:
            for member in specie.members:
                yield member


    def giveBackUp(self):
        return self.population_backup

    def giveBackUpSpecies(self):
        return self.species_backup
        
    def _evolve(self, evaluator, solution=None):
        """ A single evolutionary step .
        """
        # Unpack species
        pop = list(self.population)

        # receive the best individuals from all the other robots and choose the best, add the best as a new species
        remote_genotypes = self.distributor.get_genotypes()
        if len(remote_genotypes) is not 0:
            best_remote_ind = max(remote_genotypes, key=lambda ind: ind.stats['fitness'])
            self.species.append(NEATSpecies(best_remote_ind))
            pop.append(best_remote_ind)

        ## INITIAL BIRTH
        while len(pop) < self.popsize:
            individual = self.geno_factory()
            pop.append(individual)
            
        ## EVALUATE 
        pop = self._evaluate_all(pop, evaluator)



        self.population_backup = pop
        self.species_backup = self.species

        ## SPECIATE
        # Select random representatives
        for specie in self.species:
            specie.representative = random.choice(specie.members)
            specie.members = []
            specie.age += 1
        # Add all individuals to a species
        for individual in pop:
            found = False
            for specie in self.species:
                if individual.distance(specie.representative) <= self.current_compatibility_threshold:
                    specie.members.append(individual)
                    found = True
                    break
            # Create a new species
            if not found:
                s = NEATSpecies(individual)
                self.species.append(s)
        
        # Remove empty species
        self.species = filter(lambda s: len(s.members) > 0, self.species)
        
        # Ajust compatibility_threshold
        if len(self.species) < self.target_species:
            self.current_compatibility_threshold -= self.compatibility_threshold_delta
        elif len(self.species) > self.target_species:
            self.current_compatibility_threshold += self.compatibility_threshold_delta
        
        ## FIND CHAMPION / CHECK FOR SOLUTION
        self._find_best(pop, solution)

        # Send the best individual to all the other robots
        best_ind = self.champions[-1]
        self.distributor.send_genotype(best_ind)

        ## REPRODUCE
        
        for specie in self.species:
            specie.max_fitness_prev = specie.max_fitness
            specie.avg_fitness = np.mean( [ind.stats['fitness'] for ind in specie.members] )
            specie.max_fitness = np.max( [ind.stats['fitness'] for ind in specie.members] )
            if specie.max_fitness <= specie.max_fitness_prev:
                specie.no_improvement_age += 1
            else:
                specie.no_improvement_age = 0
            specie.has_best = self.champions[-1] in specie.members
        
        # Remove stagnated species
        # This is implemented as in neat-python, which resets the
        # no_improvement_age when the average increases
        self.species = filter(lambda s: s.no_improvement_age < self.stagnation_age or s.has_best, self.species)
        
        # Average fitness of each species
        avg_fitness = np.array([specie.avg_fitness for specie in self.species])

        # Adjust based on age
        age = np.array([specie.age for specie in self.species])
        for specie in self.species:
            if specie.age < self.young_age:
                specie.avg_fitness *= self.young_multiplier
            if specie.age > self.old_age:
                specie.avg_fitness *= self.old_multiplier

        # Compute offspring amount
        total_average = sum(specie.avg_fitness for specie in self.species)
        for specie in self.species:
            specie.offspring = int(round(self.popsize * specie.avg_fitness / total_average))
        
        # Remove species without offspring
        self.species = filter(lambda s: s.offspring > 0, self.species)
        
        # Produce offspring
        # Stanley says he resets the innovations each generation, but
        # neat-python keeps a global list.
        # This switch controls which behavior to simulate.
        if self.reset_innovations:
            self.innovations = dict()
        for specie in self.species:
            # First we keep only the best individuals of each species
            specie.members.sort(key=lambda ind: ind.stats['fitness'], reverse=True)
            keep = max(1, int(round(len(specie.members) * self.survival)))
            pool = specie.members[:keep]
            # Keep one if elitism is set and the size of the species is more than
            # that indicated in the settings
            if self.elitism and len(specie.members) > self.min_elitism_size:
                specie.members = specie.members[:1]
            else:
                specie.members = []
            # Produce offspring:
            while len(specie.members) < specie.offspring:
                # Perform tournament selection
                k = min(len(pool), self.tournament_selection_k)
                p1 = max(random.sample(pool, k), key=lambda ind: ind.stats['fitness'])
                p2 = max(random.sample(pool, k), key=lambda ind: ind.stats['fitness'])
                # Mate and mutate
                child = p1.mate(p2)
                child.mutate(innovations=self.innovations, global_innov=self.global_innov)
                specie.members.append(child)
        
        if self.innovations:
            self.global_innov = max(self.innovations.itervalues())            
        
        self._gather_stats(pop)

    def _status_report(self):
        """ Prints a status report """
        print "\n== Generation %d ==" % self.generation
        print "Best (%.2f): %s %s" % (self.champions[-1].stats['fitness'], self.champions[-1], self.champions[-1].stats)
        print "Solved: %s" % (self.solved_at)
        print "Species: %s" % ([len(s.members) for s in self.species]) 
        print "Age: %s" % ([s.age for s in self.species])
        print "No improvement: %s" % ([s.no_improvement_age for s in self.species])

        self.logger.info("== Generation %d ==" % self.generation)
        self.logger.info("Best (%.2f): %s %s" % (self.champions[-1].stats['fitness'], self.champions[-1], self.champions[-1].stats))
        self.logger.info("Solved: %s" % self.solved_at)
        self.logger.info("Species: %s" % ([len(s.members) for s in self.species]))
        self.logger.info("Age: %s" % ([s.age for s in self.species]))
        self.logger.info("No improvement: %s" % ([s.no_improvement_age for s in self.species]))


class Distributor:
    PORT = 4242

    def __init__(self, ip_address, logger):
        self.__msg_senders = dict()
        self.__msg_receivers = dict()
        self.__stopSockets = list()
        self.__address = ip_address
        self.__bots = self.get_bots()
        self.__logger = logger
        self.__inbox = Inbox(self.__logger)
        self.__started = False
        for bot in self.__bots:
            self.__logger.debug(ip_address + " added sender and receiver for: " + bot)
            self.__msg_senders[bot] = MessageSender(bot, self.PORT, self.__logger)
            self.__msg_receivers[bot] = MessageReceiver(bot, self.__inbox, self.__logger)
        self.__connection = ConnectionsListener(self.__address, self.PORT, self.__msg_receivers, self.__logger)

    def get_bots(self):
        with open('bots.txt') as bots_file:
            bots = bots_file.readlines()
        bots = filter(lambda b: b != "\n" and not b.startswith(self.__address), bots)
        return [bot.replace("\n", "") for bot in bots]

    """
        Sends the Genotype to all robots listed in te bots.txt file. The function will return True when the sending was
        successful and False otherwise
    """
    def send_genotype(self, genotype):
        self.__logger.debug("Sending genotype to listeners: " + str(genotype))
        if self.__started:
            for addr in self.__msg_senders:
                bot = self.__msg_senders[addr]
                self.__logger.debug("Sending message to " + addr)
                bot.outboxAppend(genotype)
            return True
        else:
            self.__logger.debug("Attempted to send genotype without starting the Distributor")
            return False

    def get_genotypes(self):
        genotypes = self.__inbox.popAll()
        self.__logger.info("Recieved " + str(len(genotypes)) +  " genotypes")
        return genotypes

    """
        Start waiting for messages and enable the senders to send messages to the other bots
    """
    def start(self):
        self.__connection.start()

        # for i in range(0, len(self.__msg_receivers)):
        #     stop_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #     stop_socket.connect(('127.0.0.1', '1337'))
        #     self.__stopSockets.append(stop_socket)

        for addr in self.__msg_senders:
            self.__msg_senders[addr].start()
        for addr in self.__msg_receivers:
            self.__msg_receivers[addr].start()
        self.__started = True

    def stop(self):
        if not self.__started:
            return

        for addr in self.__msg_senders:
            self.__logger.info('Killing Sender ' + addr)
            self.__msg_senders[addr].stop()
            self.__msg_senders[addr].join()
        self.__logger.info('All MessageSenders: KILLED')

        # Stopping connections listener: no more incoming connections
        self.__connection.stop()
        self.__connection.join()
        self.__logger.info('ConnectionsListener: KILLED')

        for addr in self.__msg_receivers:
            self.__logger.info("Killing receiver " + addr)
            self.__msg_receivers[addr].stop()
            self.__msg_receivers[addr].join()

        self.__logger.info("All MessageReceivers KILLED")
        self.__started = False
        self.__logger.close()


class Logger:
    def __init__(self):
        self.date = time.strftime("%d-%m-%y_%H-%M")
        # self.__logger = open("logs/" + date + "_odneat", "w")

    def __write__(self, msg):
        with open("logs/" + self.date + "_odneat", "a") as log:
            log.write(msg + "\n")

    def debug(self, msg):
        self.__write__("debug: " + msg)

    def critical(self, msg):
        self.__write__("!!CRITICAL!!: " + msg)

    def info(self, msg):
        self.__write__("info: " + msg)

    def warning(self, msg):
        self.__write__("warning!: " + msg)

    def close(self):
        self.__logger.close()