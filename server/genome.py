import os
from random import choice, random

import numpy as np
# import matplotlib.pyplot as plt

import neat
# from neat.activations import ActivationFunctionSet
# from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.config import ConfigParameter, write_pretty_params
# from neat.six_util import iteritems, iterkeys

from genes import FormNodeGene, FormConnectionGene
# from FormNetwork import FormNetwork, Scene, Mesh


class FormGenome(neat.genome.DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = FormNodeGene
        param_dict['connection_gene_type'] = FormConnectionGene
        config = neat.genome.DefaultGenomeConfig(param_dict)
        return config

    def get_new_node_key(self):
        new_id = 0
        while new_id in self.nodes:
            new_id += 1
        return new_id

    def mutate_add_node(self, config):
        if not self.connections:
            return None, None

        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = self.get_new_node_key()
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False

        i, o = conn_to_split.key
        self.add_connection(config, i, new_node_id)
        self.add_connection(config, new_node_id, o)

    def add_connection(self, config, input_key, output_key):
        # TODO: Add further validation of this connection addition?
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        # assert output_key >= 0
        # assert isinstance(enabled, bool)
        key = (input_key, output_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config)

        # connection.weight = weight
        # connection.copy = False
        # connection.enabled = True

        self.connections[key] = connection
