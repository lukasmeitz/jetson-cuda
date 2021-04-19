
import numpy as np
from numba import jit
from scipy.special import comb

from modules.line_matcher.define_transform import define_transformation, transform_modelline_batch, calc_line_distance, \
    calc_min_pair_distance


'''
# RANSAC gross architecture:
input:      reference_data  - vector of features from the left image (points or lines)
            match_data      - vector of features from right image (same feature type as first)
output:     matches         - vector of tuples [(FeatureIndexLeft, FeatureIndexRight), ... ]

'''

def randomised_ransac(model_line_pairs, scene_line_pairs):

    # Ransac parameters
    ransac_iterations = 200  # number of iterations
    ransac_threshold = 15  # threshold
    center_point = [256, 256]

    # ransac base line data
    current_best_inliers = 0
    current_best_inliers_indices = []
    current_best_transformation = []

    # generate an educated guess on how many iterations
    # probability to choose an inlier from modellines
    p_pick_inlier = 1 / len(scene_line_pairs)
    print(p_pick_inlier)

    # iterations needed
    p_estimate = p_pick_inlier ** 2
    k = np.log(1 - (0.999 * p_pick_inlier)) / np.log(1 - (p_estimate * p_pick_inlier))
    ransac_iterations = int(k) * 10

    print("proposed iterations: " + str(k))

    # generate random value vector (uniform sampled)
    random_sample_indices = np.random.rand(ransac_iterations, 2)
    random_sample_indices *= [len(scene_line_pairs) - 1, len(model_line_pairs) - 1]
    random_sample_indices = np.round(random_sample_indices)

    # perform RANSAC iterations
    for it in range(ransac_iterations):

        # pick a random pair
        sample_model_line_pair = model_line_pairs[int(np.floor(random_sample_indices[it][1]))]
        sample_scene_line_pair = scene_line_pairs[int(np.floor(random_sample_indices[it][0]))]

        # find the transformation for these points
        t = define_transformation([sample_model_line_pair[0], sample_model_line_pair[1],
                                   sample_scene_line_pair[0], sample_scene_line_pair[1]],
                                  center_point)

        # preliminary test
        preliminary_scene_line_pair_index = int(np.random.rand() * (len(scene_line_pairs) - 1))
        preliminary_model_line_pair_index = int(np.random.rand() * (len(model_line_pairs) - 1))
        transformed_sample_line = transform_modelline_batch(
                                                [model_line_pairs[preliminary_model_line_pair_index]],
                                                t[0], [t[1], t[2]], center_point)
        ml1, ml2 = transformed_sample_line[0]
        sl1, sl2 = scene_line_pairs[preliminary_scene_line_pair_index]
        error = calc_min_pair_distance(ml1, ml2, sl1, sl2)

        if error > ransac_threshold:
            continue

        print("tdd error: " + str(error))

        # convert all other model lines using this transformation
        model_lines_transformed = transform_modelline_batch(model_line_pairs, t[0], [t[1], t[2]], center_point)

        # find inliers
        inlier_index_list = []
        error_values = []
        num_inliers = 0

        for ind1 in range(len(model_lines_transformed)):
            for ind2 in range(len(scene_line_pairs)):

                # take model and scene lines
                ml1, ml2 = model_lines_transformed[ind1]
                sl1, sl2 = scene_line_pairs[ind2]

                # different error function
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

    return current_best_inliers_indices, current_best_transformation
