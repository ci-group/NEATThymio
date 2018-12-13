# This file links the thymio task with the neat libraries. This file does not need to be changed for a new task.

import base64
from json import JSONEncoder

from peas.methods.neat import NEATPopulation, NEATGenotype
import peas.methods.hyperneat as hn

import dbus
import dbus.mainloop.glib
from copy import deepcopy
import json
import time
import sys
import socket
import thread
import os
import math
from helpers import *
import object_detector
import logging
import pickle


from peas.networks import NeuralNetwork

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')
OUTPUT_PATH = os.path.join(CURRENT_FILE_PATH, 'output')
PICKLED_DIR = os.path.join(CURRENT_FILE_PATH, 'pickled')




class CustomEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert (cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        return obj.__dict__

class ThymioTask:
    def __init__(self, evaluator, use_img_client=False , OUTPUT_PATH=OUTPUT_PATH):
        self.evaluator = evaluator
        self.detector = None
        self.experiment_name = evaluator.experimentName
        self.use_odneat = '-o' in sys.argv
        self.use_hyperneat = '-h' in sys.argv
        if self.use_odneat:
            if self.use_hyperneat:
                self.method = 'odHyperNEAT'
            else:
                self.method = 'odNEAT'
        else:
            if self.use_hyperneat:
                self.method = 'HyperNEAT'
            else:
                self.method = 'NEAT'
        print 'Starting ' + self.method
        self.experiment_name = self.method + '_' + evaluator.experimentName
        self.use_img_client = use_img_client
        self.ip = sys.argv[1]
        self.commit_sha = sys.argv[2]
        mkdir_p(os.path.join(OUTPUT_PATH, self.experiment_name))
        mkdir_p(PICKLED_DIR)
        self.firstdate = None
        
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('simulationLogger')
        
        self.expertBoolean = False
        with open('bots_expert.txt', 'r') as f:
                data = f.read()
                data = data.splitlines()
                for line in data:
                    if str(line) == str(self.ip):
                        self.expertBoolean = True
        

        if self.use_hyperneat:
            evaluator._use_hyperneat = True

    def start(self, evaluator, popsize, generations, max_motor_speed, foraging):
        if self.use_hyperneat:
            # HYPERNEAT NOT USED
            genotype = lambda innovations={}: NEATGenotype()
        else:
            genotype = lambda innovations={}: NEATGenotype(inputs=task.inputs,
                                                           outputs=task.outputs,
                                                           weight_range=task.weight_range,
                                                           types=task.types,
                                                           innovations=innovations,
                                                           feedforward=task.feedforward,
                                                           prob_add_node=task.prob_add_node,
                                                           prob_add_conn=task.prob_add_conn,
                                                           prob_mutate_weight=task.prob_mutate_weight,
                                                           prob_reset_weight=task.prob_reset_weight,
                                                           prob_reenable_conn=task.prob_reenable_conn,
                                                           prob_disable_conn=task.prob_disable_conn,
                                                           prob_reenable_parent=task.prob_reenable_parent,
                                                           prob_mutate_bias=task.prob_mutate_bias,
                                                           prob_mutate_response=task.prob_mutate_response,
                                                           prob_mutate_type=task.prob_mutate_type,
                                                           stdev_mutate_weight=task.stdev_mutate_weight,
                                                           stdev_mutate_bias=task.stdev_mutate_bias,
                                                           stdev_mutate_response=task.stdev_mutate_response,
                                                           phys_dis_neat=task.phys_dis_neat,
                                                           max_depth=task.max_depth,
                                                           max_nodes=task.max_nodes,
                                                           response_default=task.response_default,
                                                           initial_weight_stdev=task.initial_weight_stdev,
                                                           bias_as_node=task.bias_as_node,
                                                           distance_excess=task.distance_excess,
                                                           distance_disjoint=task.distance_disjoint,
                                                           distance_weight=task.distance_weight,
                                                           ip_address=self.ip
                                                           )
        
        pop = NEATPopulation(genotype,
                             popsize=evaluator.popsize,
                             elitism=evaluator.elitism,
                             compatibility_threshold=evaluator.compatibility_threshold,
                             compatibility_threshold_delta=evaluator.compatibility_threshold_delta,
                             target_species=evaluator.target_species,
                             min_elitism_size=evaluator.min_elitism_size,
                             young_age=evaluator.young_age,
                             prob_mutate = evaluator.prob_mutate,
                             young_multiplier=evaluator.young_multiplier,
                             stagnation_age=evaluator.stagnation_age,
                             old_age=evaluator.old_age,
                             old_multiplier=evaluator.old_multiplier,
                             tournament_selection_k=evaluator.tournament_selection_k,
                             reset_innovations=evaluator.reset_innovations,
                             survival=evaluator.survival,
                             phys_dis_neat=evaluator.phys_dis_neat,
                             sim_dis_neat=evaluator.sim_dis_neat,
                             ip_address=self.ip
                             )

        log = {'parameters': {}, 'generations': []}

        # log neat settings.
        log['parameters'] = {
            'max_speed': max_motor_speed,
            'inputs':evaluator.inputs,
            'outputs': evaluator.outputs,
            'weight_range':evaluator.weight_range,
            'types': evaluator.types,
            'feedforward':evaluator.feedforward,
            'prob_add_node':evaluator.prob_add_node,
            'prob_add_conn':evaluator.prob_add_conn,
            'prob_mutate_weight':evaluator.prob_mutate_weight,
            'prob_reset_weight':evaluator.prob_reset_weight,
            'prob_reenable_conn':evaluator.prob_reenable_conn,
            'prob_disable_conn':evaluator.prob_disable_conn,
            'prob_reenable_parent':evaluator.prob_reenable_parent,
            'prob_mutate_bias':evaluator.prob_mutate_bias,
            'prob_mutate_response':evaluator.prob_mutate_response,
            'prob_mutate_type':evaluator.prob_mutate_type,
            'stdev_mutate_weight':evaluator.stdev_mutate_weight,
            'stdev_mutate_bias':evaluator.stdev_mutate_bias,
            'stdev_mutate_response':evaluator.stdev_mutate_response,
            'phys_dis_neat':evaluator.phys_dis_neat,
            'max_depth':evaluator.max_depth,
            'max_nodes':evaluator.max_nodes,
            'response_default':evaluator.response_default,
            'initial_weight_stdev':evaluator.initial_weight_stdev,
            'bias_as_node':evaluator.bias_as_node,
            'distance_excess':evaluator.distance_excess,
            'distance_disjoint':evaluator.distance_disjoint,
            'distance_weight':evaluator.distance_weight,
            'popsize':evaluator.popsize,
            'elitism':evaluator.elitism,
            'compatibility_threshold':evaluator.compatibility_threshold,
            'compatibility_threshold_delta':evaluator.compatibility_threshold_delta,
            'target_species':evaluator.target_species,
            'min_elitism_size':evaluator.min_elitism_size,
            'young_age':evaluator.young_age,
            'prob_mutate ': evaluator.prob_mutate,
            'young_multiplier':evaluator.young_multiplier,
            'stagnation_age':evaluator.stagnation_age,
            'old_age':evaluator.old_age,
            'old_multiplier':evaluator.old_multiplier,
            'tournament_selection_k':evaluator.tournament_selection_k,
            'reset_innovations':evaluator.reset_innovations,
            'survival':evaluator.survival,
            'phys_dis_neat':evaluator.phys_dis_neat,
            'sim_dis_neat':evaluator.sim_dis_neat,
            'ip_address':self.ip
        }
        log['parameters'].update(self.evaluator.logs())

        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SessionBus()
        thymioController = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'),
                                          dbus_interface='ch.epfl.mobots.AsebaNetwork')
        thymioController.LoadScripts(AESL_PATH, reply_handler=dbusReply, error_handler=dbusError)

        # switch thymio LEDs off
        thymioController.SendEventName('SetColor', [0, 0, 0, 0], reply_handler=dbusReply, error_handler=dbusError)
        
        # thresholds are set > SHOULD BE IN FORAGING
        if foraging == True:
            self.detector = object_detector.ObjectDetector(0.4, 0.01, thymioController)
            self.evaluator.logger.info(str(self.ip), 'Puck_threshold:' + str(self.detector.has_puck_threshold))
            self.evaluator.logger.info(str(self.ip), 'Goal_threshold:' + str(self.detector.has_goal_threshold2))

        task = self.evaluator
        task.set_thymio_controller(thymioController)

        ctrl_serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ctrl_serversocket.bind((self.ip, 1337))
        ctrl_serversocket.listen(5)


        ctrl_client = None
        img_serversocket = None
        img_client = None

        def set_client():
            global ctrl_client
            print 'Control server: waiting for socket connections...'
            (ctrl_client, address) = ctrl_serversocket.accept()
            task.set_ctrl_client(ctrl_client)
            print 'Control server: got connection from', address

        thread.start_new_thread(set_client, ())

        if self.use_img_client:
            img_serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            img_serversocket.bind((self.ip, 31337))
            img_serversocket.listen(5)

            def set_img_client():
                global img_client
                print 'Image server: waiting for socket connections...'
                (img_client, address) = img_serversocket.accept()
                print 'Image server: got connection from', address
                write_header(img_client)

            thread.start_new_thread(set_img_client, ())

        def epoch_callback(population):
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger('simulationLogger')
            # update log dump
            index=0
            population_backup = population.giveBackUp()
            species_backup = population.giveBackUpSpecies()
            generation = {'individuals': [], 'phenotypes': [], 'gen_number': population.generation}
            outputDir = os.path.join(OUTPUT_PATH, self.experiment_name)
            date = time.strftime("%d-%m-%y_%H-%M")
            PKLLog = os.path.join(outputDir, self.experiment_name + '_' + date + '_expert_' + str(population.generation) + '.pkl')
            
            for individual in population_backup:
                if index ==0:
                    index+=1
                    with open(PKLLog, 'w') as f:
                        pickle.dump(individual, f, pickle.HIGHEST_PROTOCOL)

                copied_connections = {str(key): value for key, value in individual.conn_genes.items()}
                generation['individuals'].append({
                    'node_genes': deepcopy(individual.node_genes),
                    'conn_genes': copied_connections,
                    'stats': deepcopy(individual.stats)
                })
            if self.use_hyperneat:
                for phenotype in population.phenotype_backup:
                    generation['phenotypes'].append({
                        'cm': deepcopy(phenotype.cm),
                        'act': deepcopy(phenotype.act)
                    })

            generation['species_id'] = [species.id for species in species_backup]
            generation['species_size'] = [len(species.members) for species in species_backup]
            log['generations'].append(generation)

            #task.getLogger().info(', '.join([str(ind.stats['fitness']) for ind in population_backup.population]))

            outputDir = os.path.join(OUTPUT_PATH, self.experiment_name)
            
            if population.generation == 1:
                self.firstdate = time.strftime("%d-%m-%y_%H-%M")

            date = time.strftime("%d-%m-%y_%H-%M")
    
            jsonLogFilename = os.path.join(outputDir, self.experiment_name + '_' + date + '.json')

            with open(jsonLogFilename, 'w') as f:
                #print(outputDir, self.experiment_name + '_' + date + '.json')
                json.dump(log, f, cls=CustomEncoder)
            
            
            
            # update clean file summary
            
            if self.expertBoolean == False:
                filename = os.path.join(outputDir, self.experiment_name +  '_' + self.firstdate +  '_cleanlog.txt')
                #print filename
                index=1
                for individual in population_backup:
                    msg = "%d\t%d\t%f\t%d\t%d\t%d\t%d\t%d\t%d\t" % (population.generation,index,individual.stats['fitness'],len(individual.node_genes),len(individual.conn_genes),individual.stats['specieid'],individual.stats['id'],individual.stats['parent1'],individual.stats['parent2'])
                    index+=1

                    with open(filename, 'a') as f:
                        f.write(msg + "\n")
            
                                

        try:
            evaluator = lambda evaluee: task.evaluate(NeuralNetwork(evaluee)) \
                if hasattr(evaluee, 'get_network_data') \
                else task.evaluate(evaluee)
            converter = hn.create_converter(task.substrate()) if self.use_hyperneat else lambda x: x
            pop.epoch(generations=task.generations, expertBoolean=self.expertBoolean, evaluator=evaluator, solution=evaluator, callback=epoch_callback, converter=converter)
    
        except KeyboardInterrupt:
            release_resources(task.thymioController, ctrl_serversocket, ctrl_client, img_serversocket, img_client)
            sys.exit(1)

        release_resources(task.thymioController, ctrl_serversocket, ctrl_client, img_serversocket, img_client)
        sys.exit(0)
