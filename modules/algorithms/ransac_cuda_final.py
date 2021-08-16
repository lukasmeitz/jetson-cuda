
import math
from functools import reduce

import numpy as np

from numba import cuda

from modules.components import transformation
from modules.components.evaluation import calc_min_distance
from modules.components.preprocessor import preprocessor, preprocess_length
from modules.components.transformation import define_transformation, get_transformation_cuda
from modules.handlers.load_test_sets import load_test_set
from modules.optimized import optimized_math


@cuda.jit
def iterate_2D_array(an_array, result):

    result = 999.0
    i = cuda.grid(1)

    if i < an_array.shape[0]:
        if result < an_array[i]:
            result = an_array[i]


@cuda.jit
def evaluate_line_batch(lines, reference_lines, scores, threshold):

    m, s = cuda.grid(2)
    if m < lines.shape[0] and s < reference_lines.shape[0]:

        # line1 point 1 to line2 distance
        dist_11 = math.sqrt(
            (reference_lines[s][0] - lines[m][0]) ** 2
            + (reference_lines[s][1] - lines[m][1]) ** 2)

        dist_12 = math.sqrt(
            (reference_lines[s][2] - lines[m][0]) ** 2
            + (reference_lines[s][3] - lines[m][1]) ** 2)

        if dist_11 < dist_12:
            dist_2 = math.sqrt(
                (reference_lines[s][2] - lines[m][2]) ** 2
                + (reference_lines[s][3] - lines[m][3]) ** 2)

            scores[((lines.shape[0]) * s) + m] = (dist_11 + dist_2) / 2
        else:
            dist_2 = math.sqrt(
                (reference_lines[s][0] - lines[m][2]) ** 2
                + (reference_lines[s][1] - lines[m][3]) ** 2)
            scores[((lines.shape[0]) * s) + m] = (dist_12 + dist_2) / 2

        scores[((lines.shape[0]) * s) + m] = min(scores[((lines.shape[0]) * s) + m], threshold)


@cuda.jit
def transform_line_batch(lines, rotation, transformation_distance, center_point):

    x = cuda.grid(1)
    if x < lines.shape[0]:

        # translate p1 to origin
        lines[x, 0] -= center_point[0]
        lines[x, 1] -= center_point[1]
        # rotate
        lines[x, 0] = lines[x, 0] * rotation[0, 0] + lines[x, 1] * rotation[0, 1]
        lines[x, 1] = lines[x, 0] * rotation[1, 0] + lines[x, 1] * rotation[1, 1]
        # translate forth
        lines[x, 0] += center_point[0] + transformation_distance[0]
        lines[x, 1] += center_point[1] + transformation_distance[1]

        # translate p2 to origin
        lines[x, 2] -= center_point[0]
        lines[x, 3] -= center_point[1]
        # rotate
        lines[x, 2] = lines[x, 2] * rotation[0, 0] + lines[x, 3] * rotation[0, 1]
        lines[x, 3] = lines[x, 2] * rotation[1, 0] + lines[x, 3] * rotation[1, 1]
        # translate forth
        lines[x, 2] += center_point[0] + transformation_distance[0]
        lines[x, 3] += center_point[1] + transformation_distance[1]


@cuda.reduce
def count_inlier_error(a, b):
    return a + b


def ransac_cuda_final(model_lines,
                      scene_lines,
                      random_generator,
                      center,
                      threshold=40,
                      iterations=500):

    # Parameters
    max_inliers = 0
    best_error = 99999999
    best_transformation = []
    best_matches = []

    indices = random_generator.random((iterations, 4))
    indices *= [len(model_lines) - 1,
                len(model_lines) - 1,
                len(scene_lines) - 1,
                len(scene_lines) - 1]
    indices = np.round(indices).astype(int)

    transformations = np.zeros((len(indices), 4))

    # build hypotheses
    threadsperblock = (16, 16)
    blockspergrid_x = np.math.ceil(len(indices) / threadsperblock[0])
    blockspergrid_y = np.math.ceil(len(scene_lines) / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    get_transformation_cuda[blockspergrid, threadsperblock](model_lines, scene_lines, indices, transformations)

    # loop through
    for i in range(iterations):

        if not transformations[i][0]:
            continue

        # batch transformation
        model_lines_transformed = np.copy(model_lines)
        threadsperblock = (16, 16)
        blockspergrid_x = np.math.ceil(model_lines.shape[0] / threadsperblock[0])
        blockspergrid_y = np.math.ceil(model_lines.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        rotation_matrix = np.array(((np.cos(np.radians(transformations[i][1])),
                                    -np.sin(np.radians(transformations[i][1]))),
                                    (np.sin(np.radians(transformations[i][1])),
                                     np.cos(np.radians(transformations[i][1])))))
        transform_line_batch[blockspergrid, threadsperblock](model_lines_transformed,
                                                             rotation_matrix,
                                                             np.array([transformations[i][2],
                                                                      transformations[i][2]]),
                                                             center)

        # evaluation
        results = np.zeros((len(model_lines_transformed) * len(scene_lines)))
        blockspergrid_x = np.math.ceil(len(model_lines_transformed) / threadsperblock[0])
        blockspergrid_y = np.math.ceil(len(scene_lines) / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        evaluate_line_batch[blockspergrid, threadsperblock](model_lines_transformed, scene_lines, results, threshold)

        error_cuda = count_inlier_error(results)

        if error_cuda < best_error:
            best_error = error_cuda
            best_transformation = transformations[i]

            best_matches.clear()
            max_inliers = 0

            c = 0
            for n, r in enumerate(results):

                # id generation: scores[((lines.shape[0]) * s) + m]

                # this means inlier
                if r < threshold:
                    m = c % len(model_lines)
                    s = np.floor(c / len(model_lines))
                    best_matches.append((int(m), int(s)))
                    max_inliers += 1

                c += 1

    return best_matches, best_transformation


if __name__ == "__main__":

    # prepare
    center = np.array([256, 256])
    set = 37
    seed = 2001
    rng = np.random.default_rng(seed)

    # preprocess
    scene_lines, model_lines, match_id_list = load_test_set(set, "../../")


    #scene_lines = preprocess_length(scene_lines, max_lines=300)
    #model_lines = preprocess_length(model_lines, max_lines=300)
    model_lines = np.array(model_lines)
    scene_lines = np.array(scene_lines)

    print(model_lines[0])
    print(scene_lines[0])

    # sample
    matches, transform = ransac_cuda_final(model_lines,
                                           scene_lines,
                                           rng,
                                           center,
                                           threshold=15,
                                           iterations=10000)
    print(transform)
    print(matches)
