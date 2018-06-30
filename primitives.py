import numpy as np

cube_verts = [[-0.5, -0.5, 0.5, 1],
              [0.5, -0.5, 0.5, 1],
              [-0.5, 0.5, 0.5, 1],
              [0.5, 0.5, 0.5, 1],
              [-0.5, 0.5, -0.5, 1],
              [0.5, 0.5, -0.5, 1],
              [-0.5, -0.5, -0.5, 1],
              [0.5, -0.5, -0.5, 1]]

cube_faces = [[1, 2, 4, 3], [3, 4, 6, 5], [5, 6, 8, 7],
              [7, 8, 2, 1], [2, 8, 6, 4], [7, 1, 3, 5]]

cube = (np.array(cube_verts), np.array(cube_faces, dtype='uint32'))