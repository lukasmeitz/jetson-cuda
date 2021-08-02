from modules.optimized.optimized_math import calc_distance
import numpy as np


def filter_lines(lines, max_lines=75):

    # bins
    first = []
    second = []
    third = []

    for line in lines:

        if 0 <= line[5] <= 120:
            first.append(line)
        if 120 < line[5] <= 240:
            second.append(line)
        if 240 < line[5] <= 360:
            third.append(line)

    #return_lines = take_longest(first, max_lines//3)
    #return_lines += take_longest(second, max_lines//3)
    #return_lines += take_longest(third, max_lines//3)

    return_lines = take_longest(lines, max_lines=60)

    return np.array(return_lines)


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

