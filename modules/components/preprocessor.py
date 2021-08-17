
import numpy as np
import math
from numba import cuda

from modules.algorithms.preprocessing import filter_lines
from modules.components.evaluation import calc_min_distance, calc_advanced_cross_distance, calc_angle
from modules.components.transformation import calc_distance
from modules.handlers.load_test_sets import load_test_set
from itertools import combinations


@cuda.jit
def take_pairs(model_lines, scene_lines, results, threshold):

    m, s = cuda.grid(2)

    if m < model_lines.shape[0] and s < scene_lines.shape[0]:

        # length difference
        len_ml = model_lines[m][5]
        len_sl = scene_lines[s][5]
        len_diff = np.abs((len_ml - len_sl))

        # angle
        vec_ml1 = model_lines[m][2] - model_lines[m][0]
        vec_ml2 = model_lines[m][3] - model_lines[m][1]

        vec_sl1 = scene_lines[m][2] - scene_lines[m][0]
        vec_sl2 = scene_lines[m][3] - scene_lines[m][1]

        cos_1 = vec_ml1 * vec_sl1 + vec_ml2 * vec_sl2
        cos_2 = -vec_ml2 * vec_sl1 + vec_ml1 * vec_sl2
        angle = np.degrees(np.arccos(min(abs(cos_1), 1)))

        if cos_1 * cos_2 < 0:
            angle = -angle

        # distance midpoints
        mid_ml1 = vec_ml1 / 2 + model_lines[m][0]
        mid_ml2 = vec_ml2 / 2 + model_lines[m][1]

        mid_sl1 = vec_sl1 / 2 + scene_lines[s][0]
        mid_sl2 = vec_sl2 / 2 + scene_lines[s][1]

        mid_dist = math.sqrt(
            (mid_ml1 - mid_sl1) ** 2
            + (mid_ml2 - mid_sl2) ** 2)

        if mid_dist < threshold and angle < 30 and len_diff < 20:
            results[((model_lines.shape[0]) * s) + m] = 1



def take_longest(lines, max_lines=50):

    # calculate lengths
    lengths = []
    for index, line in enumerate(lines):
        length = calc_distance(line[0:2], line[2:4])
        lengths += [[length, index]]

    # take longest line indices
    lengths = sorted(lengths, reverse=True)
    lengths = lengths[:max_lines]

    # translate to new array
    return_array = []
    for dist, index in lengths:
        return_array += [lines[index]]

    return return_array


def preprocess_length(lines, max_lines=60, debug=False):

    sorted_lines = take_longest(lines, max_lines)

    # binning
    first = []
    second = []
    third = []

    for line in sorted_lines:
        ang = line[5]
        if ang < 0:
            ang += 180
        if 0 <= ang <= 30:
            first.append(line)
        if 30 < ang <= 60:
            second.append(line)
        if 60 < ang:
            third.append(line)


    if debug:
        print("bin content after sort: " + str(len(first))
          + ", " + str(len(second))
          + ", " + str(len(third)))

    # culling of bins
    first = first[:max_lines//3]
    second = second[:max_lines//3]
    third = third[:max_lines//3]

    res = first + second + third
    return res


def get_midpoint(line):

    p1 = line[0:2]
    p2 = line[2:4]

    return ((p2 - p1) / 2) + p1


def preprocessor(model_lines, scene_lines):

    #print(model_lines[0])

    pair_indices = []

    for nm, m in enumerate(model_lines):
        for ns, s in enumerate(scene_lines):

            # length
            len1 = calc_distance(m[0:2], m[2:4])
            len2 = calc_distance(s[0:2], s[2:4])
            len_diff = np.abs((len1 - len2))

            # angle
            vec_ml1 = m[2] - m[0]
            vec_ml2 = m[3] - m[1]
            len_ml = np.sqrt((vec_ml1) ** 2 + (vec_ml2) ** 2)
            vec_ml1 /= len_ml
            vec_ml2 /= len_ml

            vec_sl1 = s[2] - s[0]
            vec_sl2 = s[3] - s[1]
            len_sl = np.sqrt((vec_sl1) ** 2 + (vec_sl2) ** 2)
            vec_sl1 /= len_sl
            vec_sl2 /= len_sl

            cos_1 = vec_ml1 * vec_sl1 + vec_ml2 * vec_sl2
            cos_2 = -vec_ml2 * vec_sl1 + vec_ml1 * vec_sl2
            angle = math.degrees(math.acos(min(abs(cos_1), 1)))

            if cos_1 * cos_2 < 0:
                angle = -angle

            if angle < -45:
                angle += 90

            if angle > 45:
                angle -= 90

            # distance midpoints
            mid_mx = vec_ml1 / 2 + m[0]
            mid_my = vec_ml2 / 2 + m[1]
            mid_sx = vec_sl1 / 2 + s[0]
            mid_sy = vec_sl2 / 2 + s[1]

            mid_dist = math.sqrt(
                (mid_sx - mid_mx) ** 2
                + (mid_sy - mid_my) ** 2)

            if s[6] == m[6] and False:
                print("correct match with angle, mid_dist, len_dif: ")
                print(angle)
                print(mid_dist)
                print(len_diff)
                print()

            if abs(angle) < 20 and mid_dist < 100 and len_diff < 100:
                pair_indices.append([nm, ns])

    return np.array(pair_indices).astype(int)


if __name__ == "__main__":

    # load data
    scene_lines, model_lines, match_id_list = load_test_set(70, "../../")

    print("max combinations before culling: " + str(len(model_lines) * len(scene_lines)))
    c = 0
    for m in model_lines:
        for s in scene_lines:
            if m[6] == s[6]:
                c += 1
    print("possible matches: " + str(c))

    # filter length
    scene_lines_lengthfiltered = filter_lines(scene_lines, max_lines=120)
    model_lines_lengthfiltered = filter_lines(model_lines, max_lines=120)

    print("max combinations after length filter: " + str(len(model_lines_lengthfiltered) * len(scene_lines_lengthfiltered)))
    c = 0
    for m in model_lines_lengthfiltered:
        for s in scene_lines_lengthfiltered:
            if m[6] == s[6]:
                c += 1
    print("possible matches after length filter: " + str(c))

    #
    # culling
    scene_lines = preprocess_length(scene_lines, max_lines=300)
    model_lines = preprocess_length(model_lines, max_lines=300)

    print("max combinations after culling: " + str(len(model_lines) * len(scene_lines)))
    c = 0
    for m in model_lines:
        for s in scene_lines:
            if m[6] == s[6]:
                c += 1
    print("possible matches after culling: " + str(c))

    #
    # pair matching
    pair_indices = preprocessor(model_lines_lengthfiltered, scene_lines_lengthfiltered)
    print("matched combinations: " + str(len(pair_indices)))

    c = 0
    for m, s in pair_indices:
        if m == s:
            c += 1
    print("correct matches: " + str(c))