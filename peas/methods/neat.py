""" This module implements the different genotypes and evolutionary
    methods used in NEAT and HyperNEAT. It is meant to be a reference
    implementation, so any inefficiencies are copied as they are
    described.
"""

### IMPORTS ###
import sys, os
import json
import logging
import random
import time
import pickle
from copy import deepcopy
from itertools import product
from collections import defaultdict

# Libs
import numpy as np
np.seterr(over='warn', divide='raise')

# Package
from .evolution import SimplePopulation
from ..networks.rnn import NeuralNetwork

# distributed neat specific
from Inbox import Inbox
from MessageSender import *
from MessageReceiver import *
from ConnectionsListener import ConnectionsListener

# Global vars
rand = random.random
current_micro_time = lambda: int(round(time.time() * 1000000))
innovation_number = lambda(inv): float(str(inv) + '.' + str(current_micro_time()))

### Classes ###
class NEATGenotype(object):
    """ Implements the NEAT genotype, consisting of node genes and connection genes.
        For parameter defaults see Stanley NEAT paper
    """
    #these are the default values, but they should be set at the experiments
    def __init__(self,
                 inputs=2,
                 outputs=1,
                 types=['tanh'],
                 topology=None,
                 feedforward=True,
                 max_depth=None,
                 max_nodes=float('inf'),
                 response_default= 1,
                 initial_weight_stdev=2.0,
                 bias_as_node=False,
                 prob_add_node=0.1,
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
                 weight_range=(-3, 3),
                 distance_excess=1.0,
                 distance_disjoint=1.0,
                 distance_weight=0.4,
                 phys_dis_neat=False,
                 ip_address=-1,
                 innovations={}):

        # Settings
        self.inputs = inputs
        self.outputs = outputs
        self.ip_address = ip_address
        self.phys_dis_neat = phys_dis_neat
        self.bias_as_node = bias_as_node
        self.response_default = response_default
        self.initial_weight_stdev = initial_weight_stdev

        self.id = -1            # unique identifier used to distinguish controllers  = id() used on conn nodes and genes id(node_genes) + id(conn_genes)
        self.parent1 = -1       # id of parent 1
        self.parent2 = -1       # id of parent 2
        #self.generation = -1    # created in which generation
        self.specieid = 0

        # Restrictions
        self.types = types
        self.feedforward = feedforward
        self.max_depth = max_depth
        self.max_nodes = max_nodes

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

        # node_genes is a dictionary with key node_id and value a tuple:
        # (fforder, type, bias, response, layer, ancestorFrom, ancestorTo)
        # ff order is to check if network is feed forward
        # ancestorFrom and ancestorTo indicate the extremes of the connection
        # in which the node is created through mutation. For the initial nodes
        # of the network, they are set to -1
        # input nodes are layer 0 and output nodes are the max dpossible depth.
        self.node_genes = {}
        self.conn_genes = {} #: Tuples of (innov, from, to, weight, enabled)

        # Create initial network
        if self.bias_as_node: self.inputs += 1
        self.create_full(innovations) if topology is None else self.create_specified(topology, innovations)

    def randomNormalWithinRange(self, mean, stdev):
        return np.clip(np.random.normal(mean, stdev), self.weight_range[0], self.weight_range[1])

    def create_full(self, innovations):
        """ Create a default fully-connected genotype. """
        max_layer = sys.maxint if (self.max_depth is None) else (self.max_depth - 1)

        for i in xrange(self.inputs):
            self.node_genes[i] = [i * 1024.0, self.types[0], 0.0, self.response_default, 0, -1, -1]

        for i in xrange(self.outputs):
            self.node_genes[i+self.inputs] = [(self.inputs + i) * 1024.0, random.choice(self.types),
                                        0.0, self.response_default, max_layer, -1, -1]

        # Create connections
        for i in xrange(self.inputs):
            for j in xrange(self.inputs, self.inputs + self.outputs):
                self.add_connection(innovations, i, j, self.randomNormalWithinRange(0.0, self.initial_weight_stdev), True)

    # not used
    def create_specified(self, topology, innovations):
        """ Create a genotype as specified in topologie. """
        fr, to = zip(*topology)
        maxnode = max(max(fr), max(to))

        if maxnode + 1 < self.inputs + self.outputs:
            raise Exception("Topology (%d) contains fewer than inputs (%d) + outputs (%d) nodes." %
                (maxnode, self.inputs, self.outputs))

        for i in xrange(maxnode+1):
            # Assign layer 0 to input nodes, otherwise just an incrementing number,
            # i.e. each node is on its own layer.
            layer = 0 if i < self.inputs else i + 1
            self.node_genes[i] = [i * 1024.0, random.choice(self.types), 0.0, self.response_default, layer, -1, -1]

        for fr, to in topology:
            self.add_connection(innovations, fr, to, self.randomNormalWithinRange(0.0, self.initial_weight_stdev), True)

    def get_non_input_nodes(self):
        return {id:node for id, node in self.node_genes.iteritems() if node[4] != 0}

    def mutate(self, innovations={}, node_innovs={}):
        """ Perform a mutation operation on this genotype """
        if rand() < self.prob_add_node and len(self.node_genes) < self.max_nodes:
            self.mutate_add_node(innovations, node_innovs)

        elif rand() < self.prob_add_conn:
            self.mutate_add_connection(innovations)
        
        else:
            self.mutate_connections(node_innovs)
            self.mutate_nodes()

        for (fr, to) in self.conn_genes:
            if self.node_genes[to][4] == 0: raise Exception("Connection TO input node not allowed.")
        return self

    def mutate_add_node(self, innovations, node_innovs):
        # Get all connections that can be split (only connections that are enabled)
        possible_to_split = [(fr, to) for (fr, to) in self.conn_genes.keys() if self.conn_genes[(fr, to)][4]]

        # If there is a max depth, we cannot create an extra layer, so, we filter on (fr, to) pairs that differ at least 2 layers
        if self.max_depth is not None:
            possible_to_split = [(fr, to) for (fr, to) in possible_to_split if self.node_genes[fr][4] + 1 < self.node_genes[to][4]]

        if possible_to_split:
            split = self.conn_genes[random.choice(possible_to_split)]
            split[4] = False # Disable old connection

            fr, to, w = split[1:4]
            avg_fforder = (self.node_genes[fr][0] + self.node_genes[to][0]) * 0.5
            # Check if this innovation (adding a new node between fr and to) has already been done in the population
            if (fr, to) in node_innovs:
                # this innovation has already been done: we have to assign the same node id
                new_id = node_innovs[(fr, to)]
            else:
                # this innovation is new: we assign a new id to the node
                new_id = max(max(self.node_genes.keys()),
                            max(node_innovs.values()) if len(node_innovs) > 0 else 0) + 1
                # add the entry to the new node innovations dict
                node_innovs[(fr, to)] = new_id

            self.add_node(new_id, avg_fforder, random.choice(self.types), 0.0, self.response_default, self.node_genes[fr][4] + 1, fr, to)
            self.add_connection(innovations, fr, new_id, w)
            self.add_connection(innovations, new_id, to, 1.0)

    def mutate_add_connection(self, innovations):
        non_input_nodes = self.get_non_input_nodes()
        potential_conns = product(self.node_genes.keys(), non_input_nodes.keys())
        # Filter further connections if we're looking only for FF networks
        if self.feedforward:
            potential_conns = ((f, t) for (f, t) in potential_conns if self.node_genes[f][0] < self.node_genes[t][0])
        # Don't create intra-layer connections if there is a max_depth
        if self.max_depth is not None:
            potential_conns = ((f, t) for (f, t) in potential_conns if  self.node_genes[f][4] < self.node_genes[t][4])
        potential_conns = list(potential_conns)

        if potential_conns:
            (fr, to) = random.choice(potential_conns)
            self.add_connection(innovations, fr, to, self.randomNormalWithinRange(0, self.stdev_mutate_weight))

    def mutate_connections(self, node_innovs):
        for cg in self.conn_genes.values():
            if rand() < self.prob_mutate_weight:
                cg[3] += self.randomNormalWithinRange(0, self.stdev_mutate_weight)
                cg[3] = np.clip(cg[3], self.weight_range[0], self.weight_range[1])
            if rand() < self.prob_reset_weight:
                cg[3] = self.randomNormalWithinRange(0, self.stdev_mutate_weight)
                cg[3] = np.clip(cg[3], self.weight_range[0], self.weight_range[1])
            if rand() < self.prob_reenable_conn and ((cg[1], cg[2]) not in node_innovs):
                cg[4] = True
            if rand() < self.prob_disable_conn:
                cg[4] = False

    def mutate_nodes(self):
        # Mutate non-input nodes
        non_input_nodes = self.get_non_input_nodes()
        for node_gene in non_input_nodes.values():
            if rand() < self.prob_mutate_bias:
                node_gene[2] += np.random.normal(0, self.stdev_mutate_bias)
                node_gene[2] = np.clip(node_gene[2], self.weight_range[0], self.weight_range[1])
            if rand() < self.prob_mutate_type:
                node_gene[1] = random.choice(self.types)
            if rand() < self.prob_mutate_response:
                node_gene[3] += np.random.normal(0, self.stdev_mutate_response)

    def add_node(self, id, fforder, type, bias, response, layer, ancestorFrom, ancestorTo):
        # final check to ensure this is actually not a duplicate node
        knownAncestors = [(node_gene[5], node_gene[5]) for node_gene in self.node_genes.values()]
        if (ancestorFrom, ancestorTo) in knownAncestors:
            return

        node_gene = [fforder, type, bias, response, layer, ancestorFrom, ancestorTo]
        self.node_genes[id] = node_gene

    def add_connection(self, innovations, fr, to, weight, enabled=True):
        # Determine innovation number. Global innovations might be reset, thus must be done in this way
        if (fr, to) in innovations:
            innov = innovations[(fr, to)]
        else:
            max_g_innov = max(innovations.itervalues()) if(innovations) else 0
            max_l_innov = max(cg[0] for cg in self.conn_genes.values()) if(len(self.conn_genes)!=0) else 0
            max_innov = int(max(max_g_innov, max_l_innov))
            innov = innovations[(fr, to)] = max_innov + 1

        self.conn_genes[(fr, to)] = [innov, fr, to, weight, enabled]

    def mate(self, other):
        """ Performs crossover between this genotype and another, and returns the child """

        child = deepcopy(self)
        child.node_genes = {}
        child.conn_genes = {}

        # Select node genes from parents
        common_nodes = self.node_genes.viewkeys() & other.node_genes.viewkeys()
        non_common_nodes = self.node_genes.viewkeys() ^ other.node_genes.viewkeys()

        for node_id in common_nodes:
            ng = random.choice((self.node_genes[node_id], other.node_genes[node_id]))
            child.node_genes[node_id] = deepcopy(ng)

        for node_id in non_common_nodes:
            if node_id in self.node_genes:
                ng = self.node_genes[node_id]
            else:
                ng = other.node_genes[node_id]
            child.node_genes[node_id] = deepcopy(ng)

        if self.phys_dis_neat:
            # Index connections by from-to pairs -> should be by innov numbers?
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
        else:
            # index the connections by innov numbers
            self_conns = dict( ((c[0], c) for c in self.conn_genes.values()) )
            other_conns = dict( ((c[0], c) for c in other.conn_genes.values()) )
            maxinnov = max( self_conns.keys() + other_conns.keys() )

            for i in range(maxinnov+1):
                cg = None
                if i in self_conns and i in other_conns:
                    cg = random.choice((self_conns[i], other_conns[i]))
                    enabled = self_conns[i][4] and other_conns[i][4]
                else:
                    if i in self_conns:
                        cg = self_conns[i]
                        enabled = cg[4]
                    elif i in other_conns:
                        cg = other_conns[i]
                        enabled = cg[4]
                if cg is not None:
                    child.conn_genes[(cg[1], cg[2])] = deepcopy(cg)
                    child.conn_genes[(cg[1], cg[2])][4] = enabled or rand() < self.prob_reenable_parent

        # Filter connections that would become recursive in the new individual.
        class fr_to_count:
            fr_count = 1
            to_count = 1

        def is_feedforward(((fr, to), cg)):
            if self.phys_dis_neat:
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
                    d += 1 # Disjoint
                else:
                    e += 1 # Excess
            else:
                raise Exception("Error in distance function")

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
        # The id of our nodes do not correspond anymore to the position of the node
        # in node_genes. Thus, it is not obvious that if a network has n nodes, the id
        # of these node will be from 0 to n-1. For the moment, we convert node_genes
        # into the old representation (list of tuples). However, the matrix will not
        # store the information about the node id (is it useful?)

        index = 0
        # id_conversion is a dict to convert the id {originalID: convertedID}
        id_conversion = {}
        node_genes_list = []
        for id, node in self.node_genes.iteritems():
            node_genes_list.append(self.node_genes[id])
            id_conversion[id] = index
            index = index + 1

        # Assemble connectivity matrix
        cm = np.zeros((len(node_genes_list), len(node_genes_list)))
        cm.fill(np.nan)

        for (_, fr, to, weight, enabled) in self.conn_genes.itervalues():
            if enabled:
                cm[id_conversion[to], id_conversion[fr]] = weight
 
        # Reorder the nodes/connections
        ff, node_types, bias, response, layer, ancestorFrom, ancestorTo = zip(*node_genes_list)
        order = [i for _,i in sorted(zip(ff, xrange(len(ff))))]
        cm = cm[:,order][order,:]
        node_types = np.array(node_types)[order]
        bias = np.array(bias)[order]
        response = np.array(response)[order]


        # Then, we multiply all the incoming connection weights by the response
        cm *= np.atleast_2d(response).T
        # Finally, add the bias as incoming weights from node-0
        if self.bias_as_node:
            cm = np.hstack((np.atleast_2d(bias).T, cm))
            cm = np.insert(cm, 0, 0.0, axis=0)
            # TODO: this is a bit ugly, we duplicate the first node type for
            # bias node. It shouldn't matter though since the bias is used as an input.
            node_types = [node_types[0]] + list(node_types)

        if self.feedforward and np.triu(np.nan_to_num(cm)).any():
            import pprint
            pprint.pprint(self.node_genes_list)
            pprint.pprint(self.conn_genes)
            print ff
            print order
            print np.sign(cm)
            raise Exception("Network is not feedforward.")

        return cm, node_types

    def __str__(self):
        return '%s with %d nodes and %d connections.' % (self.__class__.__name__,
            len(self.node_genes), len(self.conn_genes))

    def visualize(self, filename):
        return NeuralNetwork(self).visualize(filename, inputs=self.inputs, outputs=self.outputs)

class NEATSpecies(object):

    def __init__(self, initial_member, identification):
        self.members = [initial_member]
        self.representative = initial_member
        self.offspring = 0
        self.age = 0
        self.avg_fitness = 0.
        self.max_fitness = 0.
        self.max_fitness_prev = 0.
        self.no_improvement_age = 0
        self.has_best = False
        self.id = identification

class NEATPopulation(SimplePopulation):
    """ A population object for NEAT, it contains the selection
        and reproduction methods.
    """
    #these parameters should be all initialised in the task file.
    def __init__(self, geno_factory,
                 popsize=100,
                 elitism = False,
                 compatibility_threshold=3.0,
                 compatibility_threshold_delta=0.4,
                 target_species=12,
                 min_elitism_size=1,
                 young_age=10,
                 young_multiplier=1.2,
                 stagnation_age=15,
                 old_age=30,
                 old_multiplier=0.2,
                 tournament_selection_k=5,
                 reset_innovations=False,
                 survival=1,
                 prob_mutate=0.25,
                 phys_dis_neat=False,
                 use_hyperneat=False,
                 ip_address=None,
                 sim_dis_neat=False,
                 sim_geno_loc='',
                 **kwargs):
        """ Initializes the object with settings,
            does not create a population yet.

            :param geno_factory: A callable (function or object) that returns
                                 a new instance of a genotype.

        """

        super(NEATPopulation, self).__init__(geno_factory, **kwargs)
        
        self.popsize = popsize
        self.elitism = elitism
        self.tournament_selection_k=tournament_selection_k
        self.compatibility_threshold = compatibility_threshold
        self.compatibility_threshold_delta = compatibility_threshold_delta
        self.target_species = target_species
        self.reset_innovations = reset_innovations
        self.survival = survival
        self.prob_mutate=prob_mutate
        self.young_age = young_age
        self.young_multiplier = young_multiplier
        self.old_age = old_age
        self.old_multiplier = old_multiplier
        self.stagnation_age = stagnation_age
        self.min_elitism_size = min_elitism_size
        self.logger = Logger(phys_dis_neat, use_hyperneat)
        
        #used for logging, offspring is created after evaluation so no moment to give back to task files.
        self.population_backup = []
        self.species_backup = []
        self.phenotype_backup = []

        self.innovations = {} # Keep track of global innovations
        self.node_innovs = {} # Keep track of the id of the new nodes added through mutations


        # discitubted neat specific parameters
        self.phys_dis_neat = phys_dis_neat
        self.sim_dis_neat = sim_dis_neat
        self.sim_geno_loc = sim_geno_loc
        
        if self.phys_dis_neat and self.sim_dis_neat:
            raise Exception('set phys_dis_neat for physical robots and sim_dis_neat for simulation. Can not be both true')

        # Do not allow to reset innovations while using distributed neat, because the innovations
        # tracking is needed for correctly receiving an external individual
        if self.phys_dis_neat and self.reset_innovations:
                raise Exception('reset_innovations is not allowed using distributed Neat. Set reset_innovations to False')

        self.ip_address = ip_address
        
        if phys_dis_neat:
            if ip_address is None:
                raise ValueError('ip_address is not supplied! NEATPopulation cannot be initialized with distributedneat')
            self.distributor = Distributor(self.ip_address, self.logger)
            self.distributor.start()

    def _reset(self):
        """ Resets the state of this population. """
        # Keep track of some history
        self.champions = []
        self.generation = 0
        self.specie_id = 0
        self.solved_at = None
        self.stats = defaultdict(list)

        # Neat specific:
        self.species = [] # List of species
        self.innovations = {} # Keep track of global innovations
        self.node_innovs = {} # Keep track of the id of the new nodes added through mutations
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

    def _evolve(self, evaluator, converter, solution=None):
        """ A single evolutionary step. """


        pop = list(self.population)
        pop = self._birth(pop)          #returns current pop if no initi is neccesary 
        
        pop = self._speciate(pop)
        pop = self._evaluate_all(pop, evaluator, converter) # evaluate all individuals
        
        
        # Create a back up for logging because after the evaluation the new individuals are immediately made
        self.population_backup = deepcopy(pop)
        self.species_backup = deepcopy(self.species)
        self.phenotype_backup = deepcopy(self.phenotypes)

        
        # Find and send best controller to other agents
        self._find_best(pop, solution)
        print "send"
        self._send()
        
        # Receive all best controllers of the other and pick one. Note: population is one larger now but corrected in creating offspring. the pop champ can't be the received one.
        #print "receive"
        
        pop = self._receive(pop)
        #print "end receive"
        
        # Speciate population and create new offspring
        self._determine_offspring()
        pop = self._produce_offspring(pop)
    
    def _evolve_expert(self, evaluator, converter, startBoolean, startExpert, solution=None):
        """ A single evolutionary step. """
        
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('simulationLogger')
        
        pop = list(self.population)

        if startBoolean == True and startExpert == False and len(pop)>0: # change to trigger

            pop = self._evaluate_all(pop, evaluator, converter) # evaluate all individuals

        if startExpert == True:
            del pop[:]

            with open('expert1.pkl', 'r') as f:
                expert1 = pickle.load(f)
                self._convert_remote_ind(expert1, pop)
                pop = self._speciate_individual(pop,expert1)
            
            #with open('expert2.pkl', 'r') as f:
            #    expert2 = pickle.load(f)
            #    self._convert_remote_ind(expert2, pop)
            #    pop = self._speciate_individual(pop,expert2)
            
            #with open('expert3.pkl', 'r') as f:
            #    expert3 = pickle.load(f)
            #    self._convert_remote_ind(expert3, pop)
            #    pop = self._speciate_individual(pop,expert3)
            

            # run population
            pop = self._evaluate_all(pop, evaluator, converter)
            del pop[:]

        
        # Receive all best controllers of the other and pick one. Note: population is one larger now but corrected in creating offspring. the pop champ can't be the received one.
        pop = self._receive_expert(pop)



    def _birth(self, pop):
        """ If no population exists, creates new population.
            Otherwise returns current population because popsize is already full
        """
        birth_id = 0
        while len(pop) < self.popsize:
            individual = self.geno_factory(self.innovations)
            individual.id = id(individual.node_genes) + id(individual.conn_genes)
            birth_id += 1
            individual.specieid = 0
            pop.append(individual)
        
        return pop

    def _fix_network_after_conversion(self, conversion, ind):
        oldId = conversion[0]
        newId = conversion[1]
        # fix the information about the ancestors of the nodes
        for node in ind.node_genes.values():
            if node[5] == oldId:
                node[5] == newId
            if node[6] == oldId:
                node[6] == newId
        # fix the connections information
        conns_to_convert = []
        for (fr, to), conn in ind.conn_genes.iteritems():
            if fr == conversion[0] or to == conversion[0]:
                conns_to_convert.append((fr, to))

        for conn_to_convert in conns_to_convert:
            conn = list(ind.conn_genes.pop(conn_to_convert))
            conn[1] = conversion[1] if conn_to_convert[0] == conversion[0] else conn_to_convert[0]
            conn[2] = conversion[1] if conn_to_convert[1] == conversion[0] else conn_to_convert[1]
            ind.conn_genes[(conn[1], conn[2])] = conn

    def _convert_remote_ind(self, individual, pop):
        # compute the number of initial nodes of the network (input + output)
        number_initial_nodes = individual.inputs + individual.outputs
        # create a list of already converted node, to keeping track of the nodes
        # that already have the correct id. At the beginning, only the initial nodes
        # of the network have the correct id
        converted_nodes = [key for key in individual.node_genes.keys() if key < number_initial_nodes]
        # loop over the innovs of the local population, in order to assign the correct
        # id to the nodes of the remote individual. We can consider only the innovations
        # concerning the already converted (fr, to) pairs, because otherwise we could
        # consider a node with the same id that does not really match. Thus, after a new id
        # conversion, some new conversion could be performed: we need to re-iterate
        # the conversion loop until we can not convert any node
        local_node_innovs = deepcopy(self.node_innovs)

        end = False
        end_timer=0
        while not end:
            # consider only the innovations about already converted nodes
            innovs_conv = {(fr, to): id  for (fr, to), id in local_node_innovs.iteritems()
                            if fr in converted_nodes and to in converted_nodes}

            ind_nodes = individual.node_genes
            end = True
            # loop over the restricted dict of innovations
            for (fr, to), correctId in innovs_conv.iteritems():
                # store the pair (oldId, newId) that has to be converted
                toConvert = None
                for id, node in ind_nodes.iteritems():
                    # check if the innovation is about this node
                    if id != correctId and fr == node[5] and to == node[6]:
                        toConvert = (id, correctId)
                        break

                if toConvert is not None:
                    end = False
                    # perform the conversion of the node that has id=toConvert[0]
                    # into the new id toConvert[1]. We will call this node x in the
                    # comments
                    # the new id could be already an id of a node y of the network.
                    # We need to modify: that id, the connections involving that id,
                    # the ancestor data inside a node involving that id
                    if toConvert[1] in ind_nodes.keys():
                        # the new id that has to be assigned to the node x
                        # is already an id of another node y
                        # find an unused Id
                        sameFr = ind_nodes[toConvert[1]][5] == ind_nodes[toConvert[0]][5]
                        sameTo = ind_nodes[toConvert[1]][6] == ind_nodes[toConvert[0]][6]
                        samePos = sameFr and sameTo
                        if not samePos:
                            newestId = max(ind_nodes.keys() + innovs_conv.values()) + 1
                            # change the id of the node y
                            ind_nodes[newestId] = ind_nodes.pop(toConvert[1])
                            self._fix_network_after_conversion((toConvert[1], newestId), individual)

                    # Now we can convert the id of node x
                    ind_nodes[toConvert[1]] = ind_nodes.pop(toConvert[0])
                    self._fix_network_after_conversion(toConvert, individual)
                    converted_nodes.append(toConvert[1])
            # At this point, end will be True if no conversion has been performed,
            # it will be False if at least one conversion has been performed
            
        
    
        # At the end of the while loop, all the matching nodes are converted. We
        # need to ensure that all the other nodes have an id that is different
        # from all the ids of the networks of the local population, assigning a new
        # id that is unused
        max_new_local_node = max(self.node_innovs.values() + individual.node_genes.keys()) + 1
        toConvert = []
        for id, node in individual.node_genes.iteritems():
            if id not in converted_nodes:
                # we have to assign a new id to the node
                newestId = max_new_local_node
                max_new_local_node = max_new_local_node + 1
                toConvert.append((id, newestId))
        for conversion in toConvert:
            ind_nodes[conversion[1]] = ind_nodes.pop(conversion[0])
            self._fix_network_after_conversion(conversion, individual)

        # Now, all the node ids are converted. We need to accordingly fix the
        # innovation number of the connections if that innovation already happened
        # in the local population. Otherwise, we need to add that innovation to
        # the local innovs dict.
        newConnections = []
        for (fr, to), conn in individual.conn_genes.iteritems():
            if (fr, to) in self.innovations:
                connection = list(conn)
                connection[0] = self.innovations[fr, to]
                newConnections.append((fr, to, connection))
            else:
                innov = conn[0]
                # check if that innovation number is already used, if so, set a new
                # innovation number
                if innov in self.innovations.values():
                    connection = list(conn)
                    innov = max(self.innovations.values()) + 1
                    connection[0] = innov
                    newConnections.append((fr, to, connection))
                self.innovations[(fr, to)] = innov
        for connection in newConnections:
            individual.conn_genes[(connection[0], connection[1])] = connection[2]

        # Lastly, we need to add the node_innovs information about the new network
        # to the local node_innovs
        for id, node in individual.node_genes.iteritems():
            if id >= number_initial_nodes:
                # if we are considering a node added during the evolution of the
                # remote population
                if (node[5], node[6]) not in self.node_innovs and id not in self.node_innovs.values():
                    self.node_innovs[(node[5], node[6])] = id




    def _prob_lr(self, rank, u, s=2.):
        return((2-s)/u) + (2*rank*(s-1))/(u*(u-1))

    def _speciate_individual(self, pop, individual):
        """ add one individual to the correct species """
        found = False
        for specie in self.species:
            if individual.distance(specie.representative) <= self.current_compatibility_threshold:
                #print "received individual and added to existing species"
                pop.append(individual)
                individual.specieid= specie.id
                specie.members.append(individual)
                found = True
                break
            # Create a new species
        if not found:
            #print "received individual and added to new species"
            self.specie_id += 1
            s = NEATSpecies(individual, self.specie_id)
            individual.specieid=s.id
            self.species.append(s)
            pop.append(individual)

        return pop

    def _receive(self, pop):
        """ If NEAT, receives the best individuals from all the other
            robots, chooses the best, and adds the best as a new species
        """
        if self.phys_dis_neat or self.sim_dis_neat:
            remote_genotypes = self.distributor.get_genotypes() if self.phys_dis_neat else self.read_genos()
            if len(remote_genotypes) is not 0:
                fits = [ind.stats['fitness'] for ind in remote_genotypes]

                ## Fitness proportionate selection
                probs = [1. * fit / sum(fits) for fit in fits]
                probs_cum = np.cumsum(probs)
                t = (random.random() < probs_cum).tolist()
                remote_ind = remote_genotypes[t.index(True)]

                self._convert_remote_ind(remote_ind, pop)
                # add received individual to an exsiting or new specie
                #print "id received "
                #print remote_ind.id
                #print remote_ind.stats['fitness']
                pop = self._speciate_individual(pop,remote_ind)
        
        
        return pop
    
    def _receive_expert(self, pop):
        """ If NEAT, receives the best individuals from all the other
            robots, chooses the best, and adds the best as a new species
            """
        
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('simulationLogger')
        
        
        if self.phys_dis_neat or self.sim_dis_neat:
            remote_genotypes = self.distributor.get_genotypes() if self.phys_dis_neat else self.read_genos()
    
    
        if len(remote_genotypes) > 0:
                # ip_address
                
                for remote_ind in remote_genotypes:
                    found = 0
                    for individual in pop:
                        #logger.info('I have ip, o')
                        if str(remote_ind.ip_address)==str(individual.ip_address):
                            #logger.info('before')
                            #logger.info(individual.conn_genes)
                            individual.node_genes = remote_ind.node_genes
                            individual.conn_genes = remote_ind.conn_genes
                            #logger.info('after')
                            #logger.info(individual.conn_genes)
                            found=1
                    if found==0:
                        self._convert_remote_ind(remote_ind, pop)
                        pop = self._speciate_individual(pop,remote_ind)
        
        
        return pop

    def read_genos(self):
        result = []
        if not self.reset:
            folder = self.sim_geno_loc[:max([pos for pos, char in enumerate(self.sim_geno_loc) if char == "/"])]
            for root, dirs, filenames in os.walk(folder):
                for f in filenames:
                    fileLoc = (folder + '/' + f)
                    if fileLoc == self.sim_geno_loc:
                        continue
                    with open(fileLoc, 'rb') as input:
                        obj = pickle.load(input)
                        result.append(obj)
        return result




    def _speciate(self, pop):
        
        # Remove empty species
        self.species = filter(lambda s: len(s.members) > 0, self.species)
        
        """ Classifies the population into species based on existing species."""
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
                    individual.specieid=specie.id
                    found = True
                    break
            # Create a new species
            if not found:
                self.specie_id += 1
                s = NEATSpecies(individual, self.specie_id)
                individual.specieid=s.id
                self.species.append(s)


        # Remove empty species
        self.species = filter(lambda s: len(s.members) > 0, self.species)

        # Adjust compatibility_threshold
        if len(self.species) < self.target_species:
            self.current_compatibility_threshold -= self.compatibility_threshold_delta
        elif len(self.species) > self.target_species:
            self.current_compatibility_threshold += self.compatibility_threshold_delta

        return pop

    def _send(self):
         """ If neat, sends best individual to other robots """
         best_ind = self.champions[-1]

         if self.sim_dis_neat:
             with open(self.sim_geno_loc, 'wb') as output:
                 pickle.dump(best_ind, output, pickle.HIGHEST_PROTOCOL)

         if self.phys_dis_neat:
             self.distributor.send_genotype(best_ind)

    def _determine_offspring(self):
        """ For each specie, determines the amount of offspring it can create """
        champion_specie = None
        for specie in self.species:
            specie.max_fitness_prev = specie.max_fitness
            specie.avg_fitness = np.mean( [ind.stats['fitness'] for ind in specie.members] )
            specie.max_fitness = np.max( [ind.stats['fitness'] for ind in specie.members] )
            if specie.max_fitness <= specie.max_fitness_prev:
                specie.no_improvement_age += 1
            else:
                specie.no_improvement_age = 0
            specie.has_best = self.champions[-1] in specie.members
            if self.champions[-1] in specie.members:
                champion_specie = specie

        # Remove stagnated species
        # This is implemented as in neat-python, which resets the
        # no_improvement_age when the average increases
        self.species = list(filter(lambda s: s.no_improvement_age < self.stagnation_age or s.has_best, self.species))

        # Adjust based on age
        for specie in self.species:
            if specie.age < self.young_age:
                specie.avg_fitness *= self.young_multiplier
            if specie.age > self.old_age:
                specie.avg_fitness *= self.old_multiplier

        # Compute offspring amount
        total_average = sum(specie.avg_fitness for specie in self.species)
        specie_index = 0
        for s in self.species:
            s.offspring = 0
        specie = self.species[specie_index]

        # Make sure the champion specie always survives
        if self.elitism:
            champion_specie.offspring = 1

        # Apply roulette wheel selection
        roulette_wheel = specie.avg_fitness / total_average
        roulette_size = self.popsize - 1 if self.elitism else self.popsize
        #print roulette_size
        for i in range(roulette_size):
            prob = float(i)/ self.popsize
            while prob > roulette_wheel:
                specie_index += 1
                specie = self.species[specie_index]
                roulette_wheel += specie.avg_fitness / total_average
            specie.offspring += 1

        self.species = filter(lambda s: s.offspring > 0, self.species)
 

    def _produce_offspring(self,population):
        """ For each specie, produces the offspring """
        # Stanley says he resets the innovations each generation, but
        # neat-python keeps a global list.
        if self.reset_innovations:
            self.innovations = dict()
            self.node_innovs = dict()

        child_id = 0
        
        created_individuals = []

        for specie in self.species:
            # First we keep only the best individuals of each species
            specie.members.sort(key=lambda ind: ind.stats['fitness'], reverse=True)
            keep = max(1, int(round(len(specie.members) * self.survival)))
            for member in specie.members:
                member.specieid = specie.id
            pool = specie.members[:keep]
            # Keep one if elitism is set and the size of the species is more than
            # that indicated in the settings
            if self.elitism and len(specie.members) >= self.min_elitism_size:
                specie.members = specie.members[:1]
            else:
                specie.members = []
            # Produce offspring:
            created = len(specie.members)

            while created < specie.offspring:

                # Perform tournament selection
                
                k = min(len(pool), self.tournament_selection_k)
                
                p1 = max(random.sample(pool, k), key=lambda ind: ind.stats['fitness'])
                
                
                # Mutate only??
                if rand() < self.prob_mutate:
                    child = deepcopy(p1)
                    child.parent1 = p1.id
                    child.parent2 = -2
                    child.ip_address = self.ip_address
                    child_id += 1
                    child.mutate(innovations=self.innovations, node_innovs=self.node_innovs)
                    child.id = id(child.node_genes) + id(child.conn_genes)
                
                else:
                    p2 = max(random.sample(pool, k), key=lambda ind: ind.stats['fitness'])
                    # Mate and possibly mutate
                    child = p1.mate(p2)
                    child.parent1 = p1.id
                    child.parent2 = p2.id
                    child.id = child_id
                    child.ip_address = self.ip_address
                    child_id += 1
                    if rand() < self.prob_mutate:
                        child.mutate(innovations=self.innovations, node_innovs=self.node_innovs)
                    child.id = id(child.node_genes) + id(child.conn_genes)
                        
                created_individuals.append(child)

                created +=1
        
        
        population = list(self.population)
        
        for ind in created_individuals:
            population.append(ind)
        
        self._speciate(population)

        
        if len(population) != self.popsize:
            raise Exception("New offspring size is not equal to the population size")

        return population




class Distributor:
    "Distributer class for distributed neat, only used if phys_dis_neat=True in class NEATPopulation"
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
            bot = bot.rstrip() ## remove '\n' -> should have been done by get_bots(), but appearantly this didnt work
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
    def __init__(self, phys_dis_neat, use_hyperneat):
        self.date = time.strftime("%d-%m-%y_%H-%M")
        method = ("odhyperneat" if use_hyperneat else "distibuted neat") if phys_dis_neat else ("hyperneat" if use_hyperneat else "neat")
        self.fileName = "logs/" + self.date + "_" + method

    def __write__(self, msg):
        mode = 'a' if os.path.isfile(self.fileName) else 'w'
        with open(self.fileName, mode) as log:
            log.write(msg + "\n")

    def debug(self, msg):
        self.__write__("debug: " + str(msg))

    def critical(self, msg):
        self.__write__("!!CRITICAL!!: " + str(msg))

    def info(self, msg):
        self.__write__("info: " + str(msg))

    def warning(self, msg):
        self.__write__("warning!: " + str(msg))

    def close(self):
        self.__logger.close()
