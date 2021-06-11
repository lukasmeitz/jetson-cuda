from modules.optimized.optimized_math import calc_distance
import numpy as np

def filter_lines(lines, max_lines=25):

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

    return np.array(return_array)

