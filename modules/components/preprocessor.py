
import numpy as np

from modules.components.evaluation import calc_min_distance, calc_advanced_cross_distance
from modules.handlers.load_test_sets import load_test_set
from itertools import combinations

def preprocessor(lines, max_lines=30, local_thresh=50, debug=False):

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

    for f in first:
        for t in third:
            if calc_advanced_cross_distance(f, t) < local_thresh:
                pair_indices.append([f[6], t[6]])

    for s in second:
        for t in third:
            if calc_advanced_cross_distance(s, t) < local_thresh:
                pair_indices.append([s[6], t[6]])

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
