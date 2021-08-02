from modules.components.transformation import define_transformation_opencv, define_transformation_numpy, \
    define_transformation_numpy_pairs, define_transformation_pair_opencv, define_transformation_pair_midpoint_opencv, \
    define_transformation, define_perspective_transformation
from modules.handlers.load_test_sets import load_test_set
from sys import platform
import numpy as np

import timeit

def load_data(set_num):

    path = "../"

    if platform == "linux" or platform == "linux2":
        path = "/home/lukas/jetson-cuda/"

    scene_lines, model_lines, match_id_list = load_test_set(set_num, path)


    # get rid of unnecessary data
    #scene_lines = [[p1[0][0], p1[1][0], p2[0][0], p2[1][0], mid] for p1, p2, vec, mid, len, ang in scene_lines]
    #model_lines = [[p1[0][0], p1[1][0], p2[0][0], p2[1][0], mid] for p1, p2, vec, _, len, ang, _, mid, _, _, _ in
    #               model_lines]

    # give ids to lines
    scene_lines = [[line[0], line[1], line[2], line[3], num] for num, line in enumerate(scene_lines)]
    model_lines = [[line[0], line[1], line[2], line[3], num] for num, line in enumerate(model_lines)]

    return scene_lines, model_lines


def test_transformation():

    scene_lines, model_lines = load_data(2)
    #scene_lines = np.array(scene_lines)
    #model_lines = np.array(model_lines)

    m1 = define_transformation_pair_midpoint_opencv(model_lines[1], model_lines[2],
                                                    scene_lines[0], scene_lines[1])
    print(m1)
    print()

    m2 = define_transformation_numpy_pairs(model_lines[1], model_lines[2],
                                           scene_lines[0], scene_lines[1])
    print(m2)
    print()

    m3 = define_transformation(np.array([model_lines[1], model_lines[2]]),
                               np.array([scene_lines[0], scene_lines[1]]),
                               np.array([256, 256]))
    print(m3)
    print()

    m4 = define_transformation_pair_opencv(model_lines[1], model_lines[2],
                                                    scene_lines[0], scene_lines[1])
    print(m4)
    print()


    m5 = define_perspective_transformation(model_lines[1], model_lines[2],
                                    scene_lines[0], scene_lines[1])
    print(m5)
    print()


if __name__ == "__main__":

    test_transformation()