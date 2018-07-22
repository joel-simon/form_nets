from __future__ import print_function
import json
from random import choice
import neat
from genes import FormNodeGene, FormConnectionGene
from FormNetwork import FormNetwork, Scene
from genome import FormGenome

from neat.config import ConfigParameter, write_pretty_params

# Load configuration.
config = neat.Config(FormGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'server/config')

config.node_gene_type = FormNodeGene
config.connection_gene_type = FormConnectionGene

def make_population():
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    return p.population

def make_formnets():
    p = make_population()
    # print(p)
    nets = [ FormNetwork.create(g, config) for g in p.values() ]
    return nets

# g = p.population[1]

# for _ in range(5):
#     g.mutate(config.genome_config)

# visualize.draw_net(config, g, view=True, show_disabled=False)

# form = FormNetwork.create(g, config)
# form_json = form.toJSON()

 # with open(path, 'w+') as file:
 #    json.dump(result, file, indent=4)


# def save_form(form, path):
#     result = {
#         "input_nodes":  form.input_nodes,
#         "output_nodes": form.output_nodes,
#         "node_evals": []
#     }
#     for node, connections in form.node_evals:
#         node_json = {
#             "node": node,
#             "connections": []
#         }
#         for other_node, conn in connections:
#             node_json['connections'].append({
#                 'node':other_node,
#                 'enabled': conn.enabled,
#                 'transform': conn.transform.tolist()
#             })
#         result['node_evals'].append(node_json)

#     with open(path, 'w+') as file:
#         json.dump(result, file, indent=4)
#         # print(node, vars(connections))

# save_form(form, 'genome.json')
