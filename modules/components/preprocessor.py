
import numpy as np
import math
from numba import cuda

from modules.components.evaluation import calc_min_distance, calc_advanced_cross_distance
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

    print("sample line: ")
    print(lines[0])

    sorted_lines = take_longest(lines, max_lines=len(lines))

    #sorted_lines = lines[lines[:, 5].argsort()]
    #return sorted_lines[:max_lines]

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


    # culling of bins
    first = first[:max_lines//3]
    second = second[:max_lines//3]
    third = third[:max_lines//3]

    if debug:
        print("bin content after sort: " + str(len(first))
          + ", " + str(len(second))
          + ", " + str(len(third)))

    res = first + second + third
    return res


def preprocessor(lines, max_lines=60, local_thresh=50, debug=False):

    # form of input: line = [p1x, p1y, p2x, p2y, mid, angle, len]
    if debug:
        print("processing " + str(len(lines)) + " lines")
        print(lines[0])

    # space for return values
    pair_indices = []

    # sort for length
    # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
    sorted_lines = lines[lines[:, 6].argsort()]

    # binning
    first = []
    second = []
    third = []
    for line in sorted_lines:
        ang = line[5]
        if ang < 0:
            ang += 180
        if 0 <= ang <= 60:
            first.append(line)
        if 60 < ang <= 120:
            second.append(line)
        if 120 < ang:
            third.append(line)

    if debug:
        print("bin content after sort: " + str(len(first))
          + ", " + str(len(second))
          + ", " + str(len(third)))

    # culling of bins
    first = first[:max_lines//3]
    second = second[:max_lines//3]
    third = third[:max_lines//3]

    if debug:
        print("bin content after culling: " + str(len(first))
          + ", " + str(len(second))
          + ", " + str(len(third)))

    # combination of lines
    for f in first:
        for s in second:
            if calc_advanced_cross_distance(f, s) < local_thresh:
                pair_indices.append([f[6], s[6]])
                break

    for f in first:
        for t in third:
            if calc_advanced_cross_distance(f, t) < local_thresh:
                pair_indices.append([f[6], t[6]])
                break

    for s in second:
        for t in third:
            if calc_advanced_cross_distance(s, t) < local_thresh:
                pair_indices.append([s[6], t[6]])
                break

    if len(pair_indices) < 10:
        pair_indices = list(combinations(range(len(lines)), 2))
        pass

    if debug:
        print("found " + str(len(pair_indices)) + " pairs")
        print(pair_indices)

    return pair_indices


if __name__ == "__main__":

    scene_lines, model_lines, match_id_list = load_test_set(72, "../../")
    preprocessor(scene_lines)
