import time

import numpy as np
from numba import njit, prange
from scipy.special import comb

from modules.optimized.optimized_math import define_transformation, transform_modelline_batch, calc_line_distance, \
    calc_min_pair_distance


'''
# RANSAC gross architecture:
input:      reference_data  - vector of features from the left image (points or lines)
            match_data      - vector of features from right image (same feature type as first)
output:     matches         - vector of tuples [(FeatureIndexLeft, FeatureIndexRight), ... ]

'''


def optimized_ransac(model_line_pairs, scene_line_pairs, random_generator):

    # Ransac parameters
    ransac_iterations = 200  # number of iterations
    ransac_threshold = 50  # threshold
    center_point = np.array([256, 256])

    # ransac base line data
    current_best_inliers = 0
    current_best_inliers_indices = []
    current_best_transformation = []

    # convert parameters to numpy arrays
    model_line_pairs = np.array(model_line_pairs)
    scene_line_pairs = np.array(scene_line_pairs)

    # generate an educated guess on how many iterations
    # probability to choose an inlier from modellines
    p_inlier_model = 1 #model_line_count
    p_inlier_scene = 1 / len(scene_line_pairs)
    p_pick_inlier = p_inlier_model * p_inlier_scene
    print(p_pick_inlier)

    # iterations needed
    p_estimate = p_pick_inlier # ** 2
    k = np.log(1 - 0.99) / np.log(1 - p_estimate)
    ransac_iterations = max([int(k) + 1, ransac_iterations])

    print("proposed iterations: " + str(k))

    # generate random value vector (uniform sampled)
    random_sample_indices = random_generator.random((ransac_iterations, 2))
    random_sample_indices *= [len(scene_line_pairs) - 1, len(model_line_pairs) - 1]
    random_sample_indices = np.round(random_sample_indices)

    opt_times = []
    normal_times = []

    # perform RANSAC iterations
    for it in range(ransac_iterations):

        # find the transformation for two random pairs
        t = define_transformation(model_line_pairs[int(random_sample_indices[it][1])],
                                  scene_line_pairs[int(random_sample_indices[it][0])],
                                  center_point)

        # convert all other model lines
        model_lines_transformed = transform_modelline_batch(model_line_pairs,
                                                            t[0],
                                                            np.array([t[1], t[2]]),
                                                            center_point)

        # find inliers
        inlier_index_list = []
        error_values = []
        num_inliers = 0

        for ind1 in range(len(model_lines_transformed)):
            for ind2 in range(len(scene_line_pairs)):

                # take model and scene lines
                ml1, ml2 = model_lines_transformed[ind1]
                sl1, sl2 = scene_line_pairs[ind2]

                # error function
                error = calc_min_pair_distance(ml1, ml2, sl1, sl2)
                error_values += [error]

                # check whether it's an inlier or not
                if error < ransac_threshold:
                    inlier_index_list += [(ind1, ind2, error)]
                    num_inliers += 1

        # in case a new model is better - cache it
        if num_inliers >= current_best_inliers:
            current_best_transformation = t
            current_best_inliers = num_inliers
            current_best_inliers_indices = inlier_index_list

    print(np.mean(opt_times))
    print(np.mean(normal_times))

    return current_best_inliers_indices, current_best_transformation
