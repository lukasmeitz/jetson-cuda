import numpy as np

from modules.components.batch_transformation import transform_line_batch
from modules.components.evaluation import calc_min_distance
from modules.components.transformation import define_transformation

'''
# RANSAC gross architecture:
input:      reference_data  - vector of features from the left image (points or lines)
            match_data      - vector of features from right image (same feature type as first)
output:     matches         - vector of tuples [(FeatureIndexLeft, FeatureIndexRight), ... ]

'''

def ransac_standard(model_lines, scene_lines, random_generator, center_point, ransac_threshold=40, ransac_iterations=500):

    # Ransac parameters
    current_best_inliers = 0

    # generate random value vector (uniform sampled)
    random_sample_indices = random_generator.random((ransac_iterations, 4))
    random_sample_indices *= [len(scene_lines) - 1, len(scene_lines) - 1,
                              len(model_lines) - 1, len(model_lines) - 1]
    random_sample_indices = np.round(random_sample_indices)

    # perform RANSAC iterations
    for it in range(ransac_iterations):

        # find the transformation for two random pairs
        t = define_transformation(
            np.array([model_lines[int(random_sample_indices[it][2])],
                      model_lines[int(random_sample_indices[it][3])]]),

            np.array([scene_lines[int(random_sample_indices[it][0])],
                      scene_lines[int(random_sample_indices[it][1])]]),

            center_point)


        # convert all other model lines
        model_lines_transformed = transform_line_batch(model_lines,
                                                       t[0],
                                                       np.array([t[1], t[2]]),
                                                       center_point)

        # find inliers
        inlier_index_list = []
        num_inliers = 0

        for ind1 in range(len(model_lines_transformed)):
            for ind2 in range(len(scene_lines)):

                # take model and scene lines
                ml1 = model_lines_transformed[ind1]
                sl1 = scene_lines[ind2]

                # error function
                error = calc_min_distance(ml1, sl1)

                # check whether it's an inlier or not
                if error < ransac_threshold:
                    inlier_index_list += [(model_lines_transformed[ind1][6],
                                           scene_lines[ind2][6],
                                           error)]
                    num_inliers += 1

        # in case a new model is better - cache it
        if num_inliers >= current_best_inliers:# and num_inliers > 0:
            current_best_transformation = t
            current_best_inliers = num_inliers
            current_best_matches = inlier_index_list

    return current_best_matches, current_best_transformation
