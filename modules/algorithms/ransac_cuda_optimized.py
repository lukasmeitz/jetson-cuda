
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


def ransac_cuda_optimized(model_lines,
                          scene_lines,
                          random_generator, center,
                          threshold=40,
                          iterations=500):

    # Parameters
    max_inliers = 0
    best_error = 99999999
    best_transformation = []
    best_matches = []

    # generate samples
    random_sample_indices = random_generator.random((iterations, 4))
    random_sample_indices *= [len(model_lines)-1, len(model_lines)-1,
                              len(scene_lines)-1, len(scene_lines)-1]
    random_sample_indices = np.round(random_sample_indices).astype(int)

    # loop through
    for i in range(iterations):

        # define transform
        transformation = define_transformation(
            np.array([model_lines[int(random_sample_indices[i][0])],
                      model_lines[int(random_sample_indices[i][1])]]),
            np.array([scene_lines[int(random_sample_indices[i][2])],
                      scene_lines[int(random_sample_indices[i][3])]]),
            center)

        # bail-out-test
        w = np.rad2deg(np.arccos(transformation[0][0, 0]))
        t = np.sum(np.abs(transformation[1:3]))
        if t > 100 or w > 60:
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
        blockspergrid_x = np.math.ceil(len(model_lines_transformed) / threadsperblock[0])
        blockspergrid_y = np.math.ceil(len(scene_lines) / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        evaluate_line_batch[blockspergrid, threadsperblock](model_lines_transformed, scene_lines, results, threshold)

        error_cuda = count_inlier_error(results)

        if error_cuda < best_error:
            best_error = error_cuda
            best_transformation = transformation

            best_matches.clear()
            max_inliers = 0

            c = 0
            for n, r in enumerate(results):

                # id generation: scores[((lines.shape[0]) * s) + m]

                # this means inlier
                if r < threshold:
                    m = c % len(model_lines)
                    s = np.floor(c / len(model_lines))
                    best_matches.append((model_lines[int(m)][6], scene_lines[int(s)][6]))
                    max_inliers += 1

                c += 1

    return best_matches, best_transformation


if __name__ == "__main__":

    # prepare
    center = np.array([1280/2, 720/2])  # np.array([256, 256])
    set = 70
    seed = 2000
    rng = np.random.default_rng(seed)

    # preprocess
    scene_lines, model_lines, match_id_list = load_test_set(set, "../../")
    model_lines = np.array(model_lines)
    scene_lines = np.array(scene_lines)
    scene_line_pairs = preprocessor(scene_lines, max_lines=120)
    model_line_pairs = preprocessor(model_lines, max_lines=120)

    # sample
    matches, transform = ransac_cuda_optimized(model_lines, scene_lines,
                              model_line_pairs, scene_line_pairs,
                              rng, center)
    print(matches)
