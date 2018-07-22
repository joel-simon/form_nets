# cdef extern from "triangleCube.h":
#     struct Point3:
#         float x
#         float y
#         float z

#     struct Triangle3:
#         Point3 v1
#         Point3 v2
#         Point3 v3

#     long point_triangle_intersection(Point3 p, Triangle3 t)

cdef void vsub(double[:] target, double[:] a, double[:] b):
    target[0] = a[0] - b[0]
    target[1] = a[1] - b[1]
    target[2] = a[2] - b[2]

cdef void vcross(double[:] target, double[:] a, double[:] b):
    target[0] = a[1] * b[2] - a[2] * b[1]
    target[1] = a[2] * b[0] - a[0] * b[2]
    target[2] = a[0] * b[1] - a[1] * b[0]

cdef double dot(double[:] a, double[:] b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

cdef bint side(double[:] p1, double[:] p2, double[:] p3, , double[:] p4):
    cdef double[:] = np.zeros(3)


cpdef voxelize_box(int[:] dims, double voxel_length,
                   double[:,:] vertices, short[:,:] faces):

    assert vertices.shape[0] == 8
    assert vertices.shape[1] == 3
    assert faces.shape[0] == 6
    assert faces.shape[1] == 4



# cpdef voxelize_box(double[:, :] vertices, short[:,:] faces):
    # cdef Point3 *p1
    # cdef Point3 *p2
    # cdef Point3 *p3

    pass