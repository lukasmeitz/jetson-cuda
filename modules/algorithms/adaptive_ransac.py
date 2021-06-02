
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

def adaptive_ransac(model_line_pairs, scene_line_pairs, random_generator):

    # Ransac parameters
    ransac_iterations = 20000  # number of iterations
    ransac_threshold = 50  # threshold
    center_point = [256, 256]

    # ransac base line data
    current_best_inliers = 0
    current_best_inliers_indices = []
    current_best_transformation = []
    counter = 0


    # perform RANSAC iterations
    while counter < ransac_iterations:

        print( str(counter) + " of " + str(ransac_iterations))

        # generate random value vector (uniform sampled)
        random_sample_indices = random_generator.random(2)
        random_sample_indices *= [len(scene_line_pairs) - 1, len(model_line_pairs) - 1]
        random_sample_indices = np.round(random_sample_indices)

        # pick a random pair
        sample_model_line_pair = model_line_pairs[int(random_sample_indices[1])]
        sample_scene_line_pair = scene_line_pairs[int(random_sample_indices[0])]

        # find the transformation for these points
        t = define_transformation([sample_model_line_pair[0], sample_model_line_pair[1],
                                   sample_scene_line_pair[0], sample_scene_line_pair[1]],
                                  center_point)

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
        if num_inliers > current_best_inliers:
            current_best_transformation = t
            current_best_inliers = num_inliers
            current_best_inliers_indices = inlier_index_list


            # iterations needed
            p_pick_inlier = current_best_inliers / len(model_line_pairs)
            p_estimate = p_pick_inlier ** 2
            k = np.log(1 - 0.99) / np.log(1 - p_estimate)
            ransac_iterations = k

        counter += 1

    return current_best_inliers_indices, current_best_transformation
