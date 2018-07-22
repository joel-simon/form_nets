from neat.graphs import feed_forward_layers

from genes import FormNodeGene, FormConnectionGene

from neat.six_util import itervalues

from collections import namedtuple

Mesh = namedtuple('Mesh', ['nodes', 'faces', 'weight'])

class Scene(object):
    def __init__(self, mesh_list, radius=1):
        assert type(mesh_list) is list
        assert all(type(m) is Mesh for m in mesh_list)
        self.mesh_list = mesh_list
        self.radius = radius

    def to_grid(self):
        pass

    def save_obj(self, path):
        with open(path, 'w') as file:
            node_idx = 0
            for mesh in self.mesh_list:
                id_to_idx = {}

                for i, node in enumerate(mesh.nodes):
                    file.write('v %f %f %f\n' % (node[0], node[1], node[2]))
                    id_to_idx[i] = node_idx+1
                    node_idx += 1

                for i, face in enumerate(mesh.faces):
                    id1 = id_to_idx[face[0]]
                    id2 = id_to_idx[face[1]]
                    id3 = id_to_idx[face[2]]
                    id4 = id_to_idx[face[3]]

                    file.write('f %i//%i %i//%i %i//%i %i//%i\n' % \
                               (id1,id1,id2,id2,id3,id3, id4,id4))

    def transform(self, matrix, weight_mult):
        mesh_list = []

        # matrix[:3, 3] = matrix[:3, 3] * self.radius
        # matrix[0, 0] *= self.radius
        # matrix[1, 1] *= self.radius
        # matrix[2, 2] *= self.radius
        matrix[0, 0] = max(.1, matrix[0, 0])
        matrix[1, 1] = max(.1, matrix[1, 1])
        matrix[2, 2] = max(.1, matrix[2, 2])
        # matrix[3, :3] = matrix[3, :3] * 10

        # print(matrix)

        # print()

        for mesh in self.mesh_list:
            nodes = mesh.nodes.dot(matrix.T)
            # print('122132', mesh.nodes.shape, nodes.shape)
            # print(matrix, nodes)
            mesh_list.append( Mesh(nodes, mesh.faces, mesh.weight*weight_mult) )

        return Scene(mesh_list)

    def extend(self, other):
        self.mesh_list.extend(other.mesh_list)

class FormNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals

        # print(node_evals)

        self.values = { node: Scene([]) for node in inputs+outputs }

    def activate(self, inputs):
        """ Inputs is a list of Scenes.
        """
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        # print(self.values)
        # print(inputs)

        for node, connections in self.node_evals:
            self.values[ node ] = Scene([])

            for in_node, connection in connections:
                # if not connection.enabled:
                #     continue

                weight = -1.0 if connection.negate else 1.0
                scene_in = self.values[ in_node ].transform(connection.transform, weight)

                self.values[ node ].extend( scene_in )

        print({i: len(v.mesh_list) for i, v in self.values.items()}, self.output_nodes)

        return [ self.values[i] for i in self.output_nodes ]

    @staticmethod
    def create(genome, config):
        # Gather expressed connections.
        connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)

        # print(layers)

        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[ conn_key ]
                        inputs.append((inode, cg))

                ng = genome.nodes[node]
                node_evals.append((node, inputs))

        return FormNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)

    def toJSON(self):
        result = {
            "input_nodes":  self.input_nodes,
            "output_nodes": self.output_nodes,
            "node_evals": []
        }
        for node, connections in self.node_evals:
            node_json = {
                "node": node,
                "connections": []
            }
            for other_node, conn in connections:
                node_json['connections'].append({
                    'node':other_node,
                    'enabled': conn.enabled,
                    'transform': conn.transform.tolist()
                })
            result['node_evals'].append(node_json)
        return result
