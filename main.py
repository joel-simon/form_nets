from __future__ import print_function
import neat
from genes import FormNodeGene, FormConnectionGene
from FormNetwork import FormNetwork, Scene, Mesh
from primitives import cube
from neat.config import ConfigParameter, write_pretty_params

from random import choice

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

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        form = FormNetwork.create(genome, config)
        output = form.activate(Scene([cube]))

# Load configuration.
config = neat.Config(FormGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')
# print(config.input_keys)
# print(config.output_keys)

config.node_gene_type = FormNodeGene
config.connection_gene_type = FormConnectionGene


# # Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)
g = p.population[1]

for _ in range(10):
    g.mutate(config.genome_config)

print('nodes', len(g.nodes))
print('conns', len(g.connections))

form = FormNetwork.create(g, config)
initial_scene = Scene([ Mesh(cube[0], cube[0], 1) ])
# print(initial_scene)

mesh_list = form.activate( [ initial_scene ] )[0].mesh_list
print('meshs', len(mesh_list))

for mesh in mesh_list:
    print(mesh)

# # Add a stdout reporter to show progress in the terminal.
# p.add_reporter(neat.StdOutReporter(False))

# # Run until a solution is found.
# winner = p.run(eval_genomes)

# # Display the winning genome.
# print('\nBest genome:\n{!s}'.format(winner))

# # Show output of the most fit genome against training data.
# print('\nOutput:')
# # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
# # for xi, xo in zip(xor_inputs, xor_outputs):
# #     output = winner_net.activate(xi)
# #     print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
