
from numba import cuda
import numpy as np


def get_rotation_matrix_2d(rotation_angle):

    # calculate a rotation matrix
    r = np.array(((np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))),
                  (np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle)))))

    return r


def transform_line_batch(lines_in, r, transformation_distance, center_point):

    # rotation matrix
    #r = get_rotation_matrix_2d(rotation_angle)
    lines = np.copy(lines_in)

    for i in range(len(lines)):

        # get sub arrays
        p1 = lines[i, :2]
        p2 = lines[i, 2:4]

        # rotate and translate
        p1 = p1 - center_point
        p1 = r.dot(p1)
        p1 = p1 + center_point + transformation_distance
        lines[i, :2] = p1

        p2 = p2 - center_point
        p2 = r.dot(p2)
        p2 = p2 + center_point + transformation_distance
        lines[i, 2:4] = p2

    return np.array(lines)


@cuda.jit
def transform_line_batch_cuda(lines, rotation, transformation_distance, center_point):

    x = cuda.grid(1)
    if x < lines.shape[0]:


        # translate to origin
        lines[x, 0] -= center_point[0]
        lines[x, 1] -= center_point[1]
        # rotate
        lines[x, 0] = lines[x, 0] * rotation[0, 0] + lines[x, 1] * rotation[0, 1]
        lines[x, 1] = lines[x, 0] * rotation[1, 0] + lines[x, 1] * rotation[1, 1]
        # translate forth
        lines[x, 0] += center_point[0] + transformation_distance[0]
        lines[x, 1] += center_point[1] + transformation_distance[1]


        # p2
        lines[x, 2] -= center_point[0]
        lines[x, 3] -= center_point[1]
        # rotate
        lines[x, 2] = lines[x, 2] * rotation[0, 0] + lines[x, 3] * rotation[0, 1]
        lines[x, 3] = lines[x, 2] * rotation[1, 0] + lines[x, 3] * rotation[1, 1]
        # translate forth
        lines[x, 2] += center_point[0] + transformation_distance[0]
        lines[x, 3] += center_point[1] + transformation_distance[1]