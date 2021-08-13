import math
from functools import reduce

import numpy as np

from numba import cuda

from modules.components.evaluation import calc_min_distance
from modules.components.preprocessor import preprocessor
from modules.components.transformation import define_transformation
from modules.handlers.load_test_sets import load_test_set


@cuda.jit
def iterate_2D_array(an_array, result):

    result = 999.0
    i = cuda.grid(1)

    if i < an_array.shape[0]:
        if result < an_array[i]:
            result = an_array[i]


@cuda.jit
def evaluate_line_batch(lines, reference_lines, scores, inliers, threshold):

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

            res = dist_11 + dist_2
            #scores[((lines.shape[0]) * s) + m] = (dist_11 + dist_2) / 2
        else:
            dist_2 = math.sqrt(
                (reference_lines[s][0] - lines[m][2]) ** 2
                + (reference_lines[s][1] - lines[m][3]) ** 2)

            res = dist_12 + dist_2
            #scores[((lines.shape[0]) * s) + m] = (dist_12 + dist_2) / 2

        if res < threshold:
            scores[((lines.shape[0]) * s) + m] = res
            inliers[((lines.shape[0]) * s) + m] = 1


@cuda.jit
def take_pairs(model_lines, scene_lines, results, threshold):

    m, s = cuda.grid(2)

    if m < model_lines.shape[0] and s < scene_lines.shape[0]:

        # length difference
        len_ml = model_lines[m][5]
        len_sl = scene_lines[s][5]
        len_diff = abs((len_ml - len_sl))

        # angle
        vec_ml1 = model_lines[m][2] - model_lines[m][0]
        vec_ml2 = model_lines[m][3] - model_lines[m][1]

        vec_sl1 = scene_lines[m][2] - scene_lines[m][0]
        vec_sl2 = scene_lines[m][3] - scene_lines[m][1]

        cos_1 = vec_ml1 * vec_sl1 + vec_ml2 * vec_sl2
        cos_2 = -vec_ml2 * vec_sl1 + vec_ml1 * vec_sl2
        angle = math.degrees(math.acos(min(abs(cos_1), 1)))

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

        if mid_dist < threshold and angle < 30 and len_diff < 500:
            results[((model_lines.shape[0]) * s) + m] = 1


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


@cuda.jit
def map_inlier_bool(result_matrix, threshold):

    x = cuda.grid(1)

    if x < result_matrix.shape[0]:
        if result_matrix[x] < threshold:
            result_matrix[x] = 1
        else:
            result_matrix[x] = 0


def ransac_cuda_pairmatch(model_lines, scene_lines,
                          random_generator, center,
                          threshold=40):

    # debug info
    print("max combinations for evaluation: " + str(len(model_lines) * len(scene_lines)))

    # Parameters
    iterations = 2000
    best_inliers = 0
    best_error = 99999999
    best_transformation = []
    best_matches = []

    # preprocess
    match_pairs = np.zeros(len(model_lines) * len(scene_lines))
    threadsperblock = (16, 16)
    blockspergrid_x = np.math.ceil(model_lines.shape[0] / threadsperblock[0])
    blockspergrid_y = np.math.ceil(scene_lines.shape[0] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    take_pairs[blockspergrid, threadsperblock](model_lines,
                                               scene_lines,
                                               match_pairs,
                                               150)

    pairs = []
    for n, mp in enumerate(match_pairs):
        m = n % len(model_lines)
        s = np.floor(n / len(model_lines))
        pairs.append([m, s])
    pairs = np.array(pairs).astype(int)
    print("found " + str(count_inlier_error(match_pairs)) + " pairs in simple preprocessing")
    print(match_pairs)
    print()

    # generate samples
    random_sample_indices = random_generator.random((iterations, 2))
    random_sample_indices *= [len(pairs)-1, len(pairs)-1]
    random_sample_indices = np.round(random_sample_indices).astype(int)

    # loop through
    for i in range(iterations):

        # resolve index
        current_sample_indices = random_sample_indices[i]

        first_pair_indices = pairs[current_sample_indices[0]]
        second_pair_indices = pairs[current_sample_indices[1]]

        ml1 = model_lines[first_pair_indices[0]]
        sl1 = scene_lines[first_pair_indices[1]]

        ml2 = model_lines[second_pair_indices[0]]
        sl2 = scene_lines[second_pair_indices[1]]

        # define transform
        transformation = define_transformation(
            np.array([ml1, ml2]),
            np.array([sl1, sl2]),
            center)

        # bail-out-test
        w = np.rad2deg(np.arccos(transformation[0][0, 0]))
        t = np.sum(np.abs(transformation[1:3]))
        if t > 100 or w > 30:
            continue

        # batch transformation
        model_lines_transformed = np.copy(model_lines)
        threadsperblock = (16, 16)
        blockspergrid_x = np.math.ceil(model_lines.shape[0] / threadsperblock[0])
        blockspergrid_y = np.math.ceil(model_lines.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        transform_line_batch[blockspergrid, threadsperblock](model_lines_transformed,
                                                             np.array(transformation[0]),
                                                             np.array(transformation[1:3]),
                                                             center)

        # evaluation
        results = np.zeros((len(model_lines_transformed) * len(scene_lines)))
        inliers = np.zeros((len(model_lines_transformed) * len(scene_lines)))
        blockspergrid_x = np.math.ceil(len(model_lines_transformed) / threadsperblock[0])
        blockspergrid_y = np.math.ceil(len(scene_lines) / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        evaluate_line_batch[blockspergrid, threadsperblock](model_lines_transformed,
                                                            scene_lines,
                                                            results,
                                                            inliers,
                                                            threshold)

        error_cuda = count_inlier_error(results)
        inliers_cuda = count_inlier_error(inliers)

        if inliers_cuda > best_inliers and error_cuda < best_error:
            best_inliers = inliers_cuda
            best_error = error_cuda
            best_transformation = transformation
            best_matches.clear()

            for n, r in enumerate(inliers):
                # id generation: scores[((lines.shape[0]) * s) + m]
                if r == 1:
                    m = n % len(model_lines)
                    s = np.floor(n / len(model_lines))
                    best_matches.append((int(model_lines[int(m)][6]), int(scene_lines[int(s)][6])))

            print("found " + str(best_inliers) + " inliers")

    return best_matches, best_transformation

