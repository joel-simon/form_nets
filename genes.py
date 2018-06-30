from random import choice, random, randint, gauss

import numpy as np

import neat
from neat.activations import ActivationFunctionSet
from neat.attributes import BaseAttribute, FloatAttribute, BoolAttribute, StringAttribute
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import BaseGene
from neat.six_util import iteritems, iterkeys


class MatrixAttribute(BaseAttribute):
    _config_items = {"default": [np.ndarray, None],
                     "replace_rate": [float, None],
                     "mutate_rate": [float, None],
                     "mutate_power": [float, None],
                     "max_value": [float, None],
                     "min_value": [float, None]}

    def clamp(self, value, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        return np.clip(value, min_value, max_value)

    def init_value(self, config):
        # return self.default.copy()
        return np.identity(4)

    def mutate_value(self, value, config):
        """ Mutate one value in the matrix.
        """
        mutate_rate = getattr(config, self.mutate_rate_name)

        r = random()
        if r < mutate_rate:
            mutate_power = getattr(config, self.mutate_power_name)
            i = randint(0, value.shape[0]-1)
            j = randint(0, value.shape[1]-1)
            value[i, j] += gauss(0.0, mutate_power)
            return self.clamp(value, config)

        replace_rate = getattr(config, self.replace_rate_name)

        if r < replace_rate + mutate_rate:
            return self.init_value(config)

        return value

class FormNodeGene(BaseGene):
    __gene_attributes__ = []

    def distance(self, other, config):
        return 0.0

class FormConnectionGene(BaseGene):
    __gene_attributes__ = [ MatrixAttribute('transform', default=np.identity(4)),
                            BoolAttribute('copy'),
                            BoolAttribute('negate'),
                            BoolAttribute('enabled') ]

    def distance(self, other, config):
        d = np.linalg.norm(self.transform - other.transform)
        d += self.copy != other.copy
        d += self.negate != other.negate
        d += self.enabled != other.enabled
        return d * config.compatibility_weight_coefficient

