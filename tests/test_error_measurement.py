from modules.handlers.load_test_sets import load_test_set
from sys import platform

from modules.line_matcher.define_transform import calc_min_pair_distance, transform_line_batch


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
    s1_error = []
    for i in range(4):
        scene_lines, model_lines = load_data(i+1)
        error = calc_min_pair_distance(model_lines[0], model_lines[1], scene_lines[0], scene_lines[1])
        s1_error += [error]
    print("stage 1 error: " + str(s1_error))

    # stage 2
    s2_error = []
    for i in range(8):
        scene_lines, model_lines = load_data(i + 5)
        error = calc_min_pair_distance(model_lines[0], model_lines[1], scene_lines[0], scene_lines[1])
        s2_error += [error]
    print("stage 2 error: " + str(s2_error))

    # stage 3
    s3_error = []
    for i in range(10):
        scene_lines, model_lines = load_data(2)
        scene_lines = transform_line_batch(scene_lines, 0.0, [i*2, i*2], [256, 256])
        error = calc_min_pair_distance(model_lines[0], model_lines[1], scene_lines[0], scene_lines[1])
        s3_error += [error]
    print("stage 3 error: " + str(s3_error))

    # stage 4
    s4_error = []
    for i in range(15):
        scene_lines, model_lines = load_data(2)
        scene_lines = transform_line_batch(scene_lines, i, [0, 0], [256, 256])
        error = calc_min_pair_distance(model_lines[0], model_lines[1], scene_lines[0], scene_lines[1])
        s4_error += [error]
    print("stage 4 error: " + str(s4_error))

    # stage 5
    s5_error = []
    scene_lines, model_lines = load_data(2)
    for i in range(5):
        error = calc_min_pair_distance(model_lines[0], model_lines[1], scene_lines[0], scene_lines[i+2])
        s5_error += [error]
    print("stage 5 error: " + str(s5_error))

    # stage 6
    s6_error = []
    scene_lines, model_lines = load_data(2)
    for i in range(5):
        error = calc_min_pair_distance(model_lines[0], model_lines[1], scene_lines[i+2], scene_lines[i+3])
        s6_error += [error]
    print("stage 6 error: " + str(s6_error))


if __name__ == "__main__":

    test_min_pair_distance()