import itertools
import time

from modules.handlers.load_test_sets import load_test_set
from sys import platform

from modules.optimized.optimized_math import define_transformation, transform_modelline_batch


def load_data(set_num):

    path = "../"

    if platform == "linux" or platform == "linux2":
        path = "/home/lukas/jetson-cuda/"

    scene_lines, model_lines, match_id_list = load_test_set(set_num, path)

    # get rid of unnecessary data
    scene_lines = [[p1[0][0], p1[1][0], p2[0][0], p2[1][0], mid] for p1, p2, vec, mid, len, ang in scene_lines]
    model_lines = [[p1[0][0], p1[1][0], p2[0][0], p2[1][0], mid] for p1, p2, vec, _, len, ang, _, mid, _, _, _ in
                   model_lines]

    # give ids to lines
    scene_lines = [[line[0], line[1], line[2], line[3], num] for num, line in enumerate(scene_lines)]
    model_lines = [[line[0], line[1], line[2], line[3], num] for num, line in enumerate(model_lines)]

    return scene_lines, model_lines


def test_batch_transform():

    # test steps:
    # 1) load a testset with modellines and scenelines
    # 2) pick a random sample and define a transformation
    # 3) stop time taken for transform_modellines_batch

    # 1) load data
    scene_lines, model_lines = load_data(3)
    model_line_pairs = [[l1, l2] for l1, l2 in list(itertools.combinations(model_lines, r=2))]


    # 2) find the transformation for these points
    t = define_transformation([model_lines[2], model_lines[3],
                               scene_lines[4], scene_lines[5]],
                              [256, 256])

    # 3) stop times
    opt_times = []

    # numba
    start = time.time()
    model_lines_transformed = transform_modelline_batch(model_line_pairs, t[0], [t[1], t[2]], [256, 256])
    opt_times += [time.time() - start]

    print(opt_times)


if __name__ == "__main__":

    test_batch_transform()