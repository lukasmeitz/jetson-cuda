
import numpy as np

from modules.line_matcher.define_transform import define_transformation, transform_modelline_batch, calc_line_distance, \
    calc_min_pair_distance
from modules.visuals.imaging import draw_lines, plot_image

'''
# RANSAC gross architecture:
input:      reference_data  - vector of features from the left image (points or lines)
            match_data      - vector of features from right image (same feature type as first)
output:     matches         - vector of tuples [(FeatureIndexLeft, FeatureIndexRight), ... ]

'''


def ransac_draft(model_line_pairs, scene_line_pairs):

    # Ransac parameters
    ransac_iterations = 10  # number of iterations
    ransac_threshold = 25  # threshold
    center_point = [256, 256]

    # ransac base line data
    current_best_inliers = 0

    # perform RANSAC iterations
    for it in range(ransac_iterations):

        # pick a random pair
        random_sample_index_scene = round(np.random.random() * len(scene_line_pairs)-1)
        random_sample_index_model = round(np.random.random() * len(model_line_pairs)-1)
        sample_model_line_pair = model_line_pairs[random_sample_index_model]
        sample_scene_line_pair = scene_line_pairs[random_sample_index_scene]

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
                #error = calc_line_distance(ml1, ml2, sl1, sl2)
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

        #print(min(error_values))
        #print("inliers: " + str(num_inliers))


    return current_best_inliers_indices, current_best_transformation
