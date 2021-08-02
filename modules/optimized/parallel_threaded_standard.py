import time
from threading import Thread

import numpy as np
from numba import jit, prange

from modules.optimized.optimized_math import define_transformation, transform_modelline_batch, calc_line_distance, \
    calc_min_pair_distance, transform_line_batch, calc_min_line_distance

'''
# RANSAC gross architecture:
input:      reference_data  - vector of features from the left image (points or lines)
            match_data      - vector of features from right image (same feature type as first)
output:     matches         - vector of tuples [(FeatureIndexLeft, FeatureIndexRight), ... ]

'''


def single_iteration_pass(data, id_num, center_point, ransac_threshold, results):

    #print("starting thread " + str(id_num))

    # de-reference data for readability
    model_lines = data[0]
    scene_lines = data[1]
    random_sample_indices = data[2]

    # find the transformation for two random pairs
    t = define_transformation(
        np.array([model_lines[int(random_sample_indices[id_num][2])],
                  model_lines[int(random_sample_indices[id_num][3])]]),

        np.array([scene_lines[int(random_sample_indices[id_num][0])],
                  scene_lines[int(random_sample_indices[id_num][1])]]),

        center_point)

    # convert all other model lines
    model_lines_transformed = transform_line_batch(model_lines,
                                                      t[0],
                                                      np.array([t[1], t[2]]),
                                                      center_point)


    # find inliers
    inlier_index_list = []
    num_inliers = 0

    error_avg  = 0

    for ind1 in range(len(model_lines)):
        for ind2 in range(len(scene_lines)):

            # take model and scene lines
            ml1 = model_lines_transformed[ind1]
            sl1 = scene_lines[ind2]

            # error function
            error = calc_min_line_distance(ml1, sl1)

            # check whether it's an inlier or not
            if error < ransac_threshold:
                inlier_index_list += [(ind1, ind2, error)]
                num_inliers += 1
                error_avg += error

    # write error to data structure
    results[id_num] = num_inliers

    return



def ransac_parallel_threaded(model_lines, scene_lines,
                   center_point, random_generator,
                   result_transformation, model_lines_transformed,
                   sync):

    # Ransac parameters
    ransac_iterations = 500  # number of iterations
    ransac_threshold = 75  # threshold
    current_best_inliers = 0
    min_error = 999

    # generate random value vector (uniform sampled)
    random_sample_indices = random_generator.random((ransac_iterations, 4))
    random_sample_indices *= [len(scene_lines) - 1, len(scene_lines) - 1,
                              len(model_lines) - 1, len(model_lines) - 1]
    random_sample_indices = np.round(random_sample_indices)

    data = [model_lines, scene_lines, random_sample_indices]
    threads = []
    results = np.zeros(len(random_sample_indices))

    # perform RANSAC iterations
    for it in prange(ransac_iterations):

        t = Thread(target=single_iteration_pass, args=(data, it, center_point, ransac_threshold, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    #print("all threads finished, evaluating")
    #print(results)
    #print("max inliers: " + str(max(results)))
    #return



    # in case a new model is better - cache it
    if num_inliers >= current_best_inliers and num_inliers > 0:
        current_best_transformation = t
        current_best_inliers = num_inliers
        min_error = sum(error_values)

        if sync.locked():
            sync.release()
            print(current_best_inliers)
            #print(error_values)
        time.sleep(0.3)