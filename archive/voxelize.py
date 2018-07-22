import math
import numpy as np
from pykdtree.kdtree import KDTree

def side_of(nodes, face, p):
    v1 = nodes[face[0]][:3]
    v2 = nodes[face[1]][:3]
    v3 = nodes[face[2]][:3]
    norm = np.cross(v2-v1, v3-v1)
    return np.sign(np.dot(norm, p-v1))

def point_in_box(nodes, faces, p):
    s1 = side_of(nodes, faces[0], p)
    s2 = side_of(nodes, faces[1], p)
    s3 = side_of(nodes, faces[2], p)
    s4 = side_of(nodes, faces[3], p)
    return s1 == s2 and s3 == s1 and s4 == s1

def voxelize_box(grid, mesh, dims, block_size):
    nodes = mesh.nodes
    faces = mesh.faces

    w = dims[0] * block_size * 0.5
    h = dims[1] * block_size * 0.5
    d = dims[2] * block_size * 0.5

    for i in range(dims[0]):
        for j in range(dims[0]):
            for k in range(dims[0]):
                x = i * block_size + block_size*.5 - w
                y = j * block_size + block_size*.5 - h
                z = k * block_size + block_size*.5 - d

                if point_in_box(nodes, faces, [x, y, z]):
                    grid[i, j, k] = 1

def voxelize_scene(scene, dims, block_size):
    grid = np.zeros(dims, dtype='uint8')

    for mesh in scene.mesh_list:
        voxelize_box(grid, mesh, dims, block_size)

    return grid
