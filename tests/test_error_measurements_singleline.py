from modules.handlers.load_test_sets import load_test_set
from sys import platform

from modules.line_matcher.define_transform import transform_line_batch
from modules.components.evaluation import *


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


def test_min_pair_distance():

    # tests in stages:
    # 1) lines are the same
    # 2) lines have noise on them
    #    - difference in angle between line pairs (rotate only one line)
    #    -
    # 3) lines are transformed
    # 4) lines are rotated
    # 5) one line wrong
    # 6) two lines wrong

    # stage 1
    print("stage 1: equal pairs")
    error_min = []
    error_max = []
    error_att = []
    error_cross = []
    for i in range(4):
        scene_lines, model_lines = load_data(i+1)
        error = calc_min_distance(model_lines[0], scene_lines[0])
        error_min += [error]
        error = calc_max_distance(model_lines[0], scene_lines[0])
        error_max += [error]
        error = calc_cross_distance(model_lines[0], scene_lines[0])
        error_cross += [error]
        error = calc_attribute_error(model_lines[0], scene_lines[0])
        error_att += [error]
    print("stage 1 error min distance: " + str(error_min))
    print("stage 1 error max distance: " + str(error_max))
    print("stage 1 error cross distance: " + str(error_cross))
    print("stage 1 error attribute: " + str(error_att))
    print()

    # stage 2
    print("stage 2: noise")
    error_min = []
    error_max = []
    error_att = []
    error_cross = []
    for i in range(8):
        scene_lines, model_lines = load_data(i + 5)
        error = calc_min_distance(model_lines[0],  scene_lines[0])
        error_min += [error]
        error = calc_max_distance(model_lines[0], scene_lines[0])
        error_max += [error]
        error = calc_cross_distance(model_lines[0], scene_lines[0])
        error_cross += [error]
        error = calc_attribute_error(model_lines[0], scene_lines[0])
        error_att += [error]
    print("stage 2 error min distance: " + str(error_min))
    print("stage 2 error max distance: " + str(error_max))
    print("stage 2 error cross distance: " + str(error_cross))
    print("stage 2 error attribute: " + str(error_att))
    print()

    # stage 3
    print("stage 3: translation")
    error_min = []
    error_max = []
    error_att = []
    error_cross = []
    for i in range(10):
        scene_lines, model_lines = load_data(2)
        scene_lines = transform_line_batch(scene_lines, 0.0, [i*2, i*2], [256, 256])
        error = calc_min_distance(model_lines[0], scene_lines[0])
        error_min += [error]
        error = calc_max_distance(model_lines[0], scene_lines[0])
        error_max += [error]
        error = calc_cross_distance(model_lines[0], scene_lines[0])
        error_cross += [error]
        error = calc_attribute_error(model_lines[0], scene_lines[0])
        error_att += [error]
    print("stage 3 error min distance: " + str(error_min))
    print("stage 3 error max distance: " + str(error_max))
    print("stage 3 error cross distance: " + str(error_cross))
    print("stage 3 error attribute: " + str(error_att))
    print()

    # stage 4
    print("stage 4: rotation")
    error_min = []
    error_max = []
    error_att = []
    error_cross = []
    for i in range(15):
        scene_lines, model_lines = load_data(2)
        scene_lines = transform_line_batch(scene_lines, i, [0, 0], [256, 256])
        error = calc_min_distance(model_lines[0], scene_lines[0])
        error_min += [error]
        error = calc_max_distance(model_lines[0], scene_lines[0])
        error_max += [error]
        error = calc_cross_distance(model_lines[0], scene_lines[0])
        error_cross += [error]
        error = calc_attribute_error(model_lines[0], scene_lines[0])
        error_att += [error]
    print("stage 4 error min distance: " + str(error_min))
    print("stage 4 error max distance: " + str(error_max))
    print("stage 4 error cross distance: " + str(error_cross))
    print("stage 4 error attribute: " + str(error_att))
    print()

    # stage 5
    print("stage 5: scaling of lines")
    error_min = []
    error_max = []
    error_att = []
    error_cross = []
    for i in range(10):
        scene_lines, model_lines = load_data(2)
        model_lines = np.dot(model_lines, (i * (0.2)) + 0.2)
        scene_lines = np.dot(scene_lines, (i * (0.2)) + 0.2)
        error = calc_min_distance(model_lines[0], scene_lines[0])
        error_min += [error]
        error = calc_max_distance(model_lines[0], scene_lines[0])
        error_max += [error]
        error = calc_cross_distance(model_lines[0], scene_lines[0])
        error_cross += [error]
        error = calc_attribute_error(model_lines[0], scene_lines[0])
        error_att += [error]
    print("stage 5 error min distance: " + str(error_min))
    print("stage 5 error max distance: " + str(error_max))
    print("stage 5 error cross distance: " + str(error_cross))
    print("stage 5 error attribute: " + str(error_att))
    print()


    # stage 6
    print("stage 6: lines of different length")
    error_min = []
    error_max = []
    error_att = []
    error_cross = []
    for i in range(10):
        scene_lines, model_lines = load_data(2)
        scene_lines = np.array(scene_lines)
        scene_lines[:, 2] *= (1.0 + (i*0.1))
        scene_lines[:, 3] *= (1.0 + (i*0.1))

        error = calc_min_distance(model_lines[0], scene_lines[0])
        error_min += [error]
        error = calc_max_distance(model_lines[0], scene_lines[0])
        error_max += [error]
        error = calc_cross_distance(model_lines[0], scene_lines[0])
        error_cross += [error]
        error = calc_attribute_error(model_lines[0], scene_lines[0])
        error_att += [error]
    print("stage 6 error min distance: " + str(error_min))
    print("stage 6 error max distance: " + str(error_max))
    print("stage 6 error cross distance: " + str(error_cross))
    print("stage 6 error attribute: " + str(error_att))
    print()



if __name__ == "__main__":

    test_min_pair_distance()