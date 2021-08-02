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
    s1_error_min = []
    s1_error_cross = []
    s1_error_max = []
    s1_error_att = []
    for i in range(4):
        scene_lines, model_lines = load_data(i+1)
        error = calc_min_pair_distance(model_lines[0], model_lines[1], scene_lines[0], scene_lines[1])
        s1_error_min += [error]
        error = calc_cross_distance_pairs(model_lines[0], model_lines[1], scene_lines[0], scene_lines[1])
        s1_error_cross += [error]
        error = calc_attribute_error_pairs(model_lines[0], model_lines[1], scene_lines[0], scene_lines[1])
        s1_error_att += [error]
        error = calc_max_pair_distance(model_lines[0], model_lines[1], scene_lines[0], scene_lines[1])
        s1_error_max += [error]
    print("stage 1 error min distance: " + str(s1_error_min))
    print("stage 1 error max distance: " + str(s1_error_max))
    print("stage 1 error cross distance: " + str(s1_error_cross))
    print("stage 1 error att distance: " + str(s1_error_att))
    print()

    # stage 2
    print("stage 2: noise")
    s2_error_min = []
    s2_error_cross = []
    for i in range(8):
        scene_lines, model_lines = load_data(i + 5)
        error = calc_min_pair_distance(model_lines[0], model_lines[1], scene_lines[0], scene_lines[1])
        s2_error_min += [error]
        error = calc_cross_distance_pairs(model_lines[0], model_lines[1], scene_lines[0], scene_lines[1])
        s2_error_cross += [error]
    print("stage 2 error min: " + str(s2_error_min))
    print("stage 2 error cross: " + str(s2_error_cross))
    print()

    # stage 3
    print("stage 3: translation")
    s3_error_min = []
    s3_error_cross = []
    for i in range(10):
        scene_lines, model_lines = load_data(2)
        scene_lines = transform_line_batch(scene_lines, 0.0, [i*2, i*2], [256, 256])
        error = calc_min_pair_distance(model_lines[0], model_lines[1], scene_lines[0], scene_lines[1])
        s3_error_min += [error]
        error = calc_cross_distance_pairs(model_lines[0], model_lines[1], scene_lines[0], scene_lines[1])
        s3_error_cross += [error]
    print("stage 3 error min: " + str(s3_error_min))
    print("stage 3 error cross: " + str(s3_error_cross))
    print()

    # stage 4
    print("stage 4: rotation")
    s4_error_min = []
    s4_error_cross = []
    for i in range(15):
        scene_lines, model_lines = load_data(2)
        scene_lines = transform_line_batch(scene_lines, i, [0, 0], [256, 256])
        error = calc_min_pair_distance(model_lines[0], model_lines[1], scene_lines[0], scene_lines[1])
        s4_error_min += [error]
        error = calc_cross_distance_pairs(model_lines[0], model_lines[1], scene_lines[0], scene_lines[1])
        s4_error_cross += [error]
    print("stage 4 error min: " + str(s4_error_min))
    print("stage 4 error cross: " + str(s4_error_cross))
    print()


    # stage 5
    print("stage 5: one wrong line")
    s5_error_min = []
    s5_error_max = []
    s5_error_att = []
    s5_error_cross = []
    scene_lines, model_lines = load_data(4)
    for i in range(5):
        error = calc_min_pair_distance(model_lines[0], model_lines[1], scene_lines[0], scene_lines[i+2])
        s5_error_min += [error]
        error = calc_cross_distance_pairs(model_lines[0], model_lines[1], scene_lines[0], scene_lines[i+2])
        s5_error_cross += [error]
        error = calc_attribute_error_pairs(model_lines[0], model_lines[1], scene_lines[0], scene_lines[i + 2])
        s5_error_att += [error]
        error = calc_max_pair_distance(model_lines[0], model_lines[1], scene_lines[0], scene_lines[i + 2])
        s5_error_max += [error]
    print("stage 5 error min: " + str(s5_error_min))
    print("stage 5 error max: " + str(s5_error_max))
    print("stage 5 error cross: " + str(s5_error_cross))
    print("stage 5 error att: " + str(s5_error_att))
    print()

    # stage 6
    print("stage 6: two wrong lines")
    s6_error_min = []
    s6_error_cross = []
    s6_error_max = []
    s6_error_att = []
    scene_lines, model_lines = load_data(2)
    for i in range(5):
        error = calc_min_pair_distance(model_lines[0], model_lines[1], scene_lines[i+2], scene_lines[i+3])
        s6_error_min += [error]
        error = calc_cross_distance_pairs(model_lines[0], model_lines[1], scene_lines[i+2], scene_lines[i+3])
        s6_error_cross += [error]
        error = calc_max_pair_distance(model_lines[0], model_lines[1], scene_lines[i+2], scene_lines[i+3])
        s6_error_max += [error]
        error = calc_attribute_error_pairs(model_lines[0], model_lines[1], scene_lines[i+2], scene_lines[i+3])
        s6_error_att += [error]
    print("stage 6 error min: " + str(s6_error_min))
    print("stage 6 error max: " + str(s6_error_max))
    print("stage 6 error cross: " + str(s6_error_cross))
    print("stage 6 error att: " + str(s6_error_att))
    print()


if __name__ == "__main__":

    test_min_pair_distance()