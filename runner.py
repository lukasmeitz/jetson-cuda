import datetime
import json
import os
import time
from sys import platform

from modules.algorithms.preprocessing import filter_lines
from modules.algorithms.ransac_cuda_final import ransac_cuda_final
from modules.algorithms.ransac_cuda_opencv import ransac_cuda_opencv
from modules.algorithms.ransac_cuda_optimized import ransac_cuda_optimized
from modules.algorithms.ransac_cuda_pairmatch import ransac_cuda_pairmatch
from modules.algorithms.ransac_cuda_simple import ransac_cuda_simple
from modules.algorithms.ransac_stage_1 import ransac_stage_1
from modules.algorithms.ransac_standard import ransac_standard
from modules.algorithms.ransac_standard_optimized import ransac_standard_optimized
from modules.components.batch_transformation import transform_line_batch
from modules.components.preprocessor import preprocessor, preprocess_length
from modules.handlers.load_test_sets import load_test_set
from modules.optimized.optimized_math import get_rotation_matrix_2d
from tests.test_imaging import *
from tests.test_gtm_handler import *


def do_test_run(set_number, algorithm, seed, iterations=1000, thresh=15, max_lines=120):

    print("Doing Test #" + str(set_number) + " using " + str(algorithm) + " RANSAC")

    '''
    parameters
    '''

    # dict to gather test run information
    meta = {}

    # determine platform and configure resource path
    if platform == "linux" or platform == "linux2":
        meta["path"] = "/home/lukas/jetson-cuda/"
        meta["system"] = "Jetson Board"
    else:
        meta["path"] = ""
        meta["system"] = "Windows Desktop"

    # select a single line matching test set between 1 and 45
    meta["test_set_number"] = set_number
    meta["ransac_type"] = algorithm

    # create folder structure
    dump_path = "results/" + str(algorithm) + "/Set_" + "{:03d}".format(set_number) + "/" + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S-%f")) + "/"

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    else:
        print("skipping")
        return

    rng = np.random.default_rng(seed)
    center = np.array([1280/2, 720/2]) if set_number > 50 else np.array([256, 256])

    '''
    data loading
    '''

    scene_lines, model_lines, match_id_list = load_test_set(meta["test_set_number"], meta["path"])
    meta["scene_lines"] = len(scene_lines)
    meta["model_lines"] = len(model_lines)
    meta["match_id_list_gtm"] = [int(mid)-1 for mid in match_id_list if mid != 0]
    meta["match_count_gtm"] = len(meta["match_id_list_gtm"])

    scene_lines_postprocess = scene_lines.copy()
    model_lines_postprocess = model_lines.copy()

    np.random.shuffle(scene_lines)
    np.random.shuffle(model_lines)

    '''
    algorithm
    '''

    # feed this to ransac

    if meta["ransac_type"] == "first":
        start_time = time.time()

        scene_lines_filtered = filter_lines(scene_lines, max_lines)
        model_lines_filtered = filter_lines(model_lines, max_lines)

        meta["duration_preprocess"] = (time.time() - start_time)

        matching_correspondence, transformation_2d = ransac_standard(model_lines_filtered,
                                                                     scene_lines_filtered,
                                                                     rng,
                                                                     center,
                                                                     ransac_threshold=thresh,
                                                                     ransac_iterations=iterations)
        meta["duration"] = (time.time() - start_time)

    if meta["ransac_type"] == "standard":
        start_time = time.time()

        scene_lines_filtered = preprocess_length(scene_lines, max_lines)
        model_lines_filtered = preprocess_length(model_lines, max_lines)

        meta["duration_preprocess"] = (time.time() - start_time)

        matching_correspondence, transformation_2d = ransac_standard_optimized(model_lines_filtered,
                                                                               scene_lines_filtered,
                                                                               rng, center,
                                                                               threshold=thresh,
                                                                               iterations=iterations)
        meta["duration"] = (time.time() - start_time)

    elif meta["ransac_type"] == "cuda":
        start_time = time.time()

        scene_lines_filtered = preprocess_length(scene_lines, max_lines)
        model_lines_filtered = preprocess_length(model_lines, max_lines)
        scene_lines_filtered = np.array(scene_lines_filtered)
        model_lines_filtered = np.array(model_lines_filtered)

        meta["duration_preprocess"] = (time.time() - start_time)

        matching_correspondence, transformation_2d = ransac_cuda_optimized(model_lines_filtered,
                                                                           scene_lines_filtered,
                                                                           rng, center,
                                                                           threshold=thresh,
                                                                           iterations=iterations)
        meta["duration"] = (time.time() - start_time)

    elif meta["ransac_type"] == "final":
        start_time = time.time()

        scene_lines_filtered = preprocess_length(scene_lines, max_lines)
        model_lines_filtered = preprocess_length(model_lines, max_lines)
        scene_lines_filtered = np.array(scene_lines_filtered)
        model_lines_filtered = np.array(model_lines_filtered)

        meta["duration_preprocess"] = (time.time() - start_time)

        matching_correspondence, transformation_2d = ransac_cuda_final(model_lines_filtered,
                                                                       scene_lines_filtered,
                                                                       rng, center,
                                                                       threshold=thresh,
                                                                       iterations=iterations)
        meta["duration"] = (time.time() - start_time)

    elif meta["ransac_type"] == "pairwise":
        start_time = time.time()

        scene_lines_filtered = np.array(preprocess_length(scene_lines, max_lines))
        model_lines_filtered = np.array(preprocess_length(model_lines, max_lines))

        meta["duration_preprocess"] = (time.time() - start_time)

        matching_correspondence, transformation_2d = ransac_cuda_pairmatch(model_lines_filtered,
                                                                           scene_lines_filtered,
                                                                           rng,
                                                                           center,
                                                                           thresh,
                                                                           iterations)
        meta["duration"] = (time.time() - start_time)


    '''
    post processing
    '''

    meta["match_id_list"] = matching_correspondence

    # get true positives
    match_positives = 0
    for c in matching_correspondence:
        if c[0] == c[1]:
            match_positives += 1
    meta["match_count"] = match_positives
    meta["negative_count"] = len(matching_correspondence) - match_positives

    #print(transformation_2d)
    #meta["transformation"] = transformation_2d

    with open(meta["path"] + dump_path + "matching_result" + str(meta["test_set_number"]) + ".json", 'w') as fp:
        json.dump(meta, fp)

    print("found " + str(meta["match_count"]) + " of " + str(meta["match_count_gtm"]) + " matches")
    print("found " + str(meta["negative_count"]) + " false matches")
    print("took " + str(meta["duration"]) + " seconds")
    print("took " + str(meta["duration_preprocess"]) + " for preprocessing")
    print()

    '''
    visualization
    '''
    image_size = center * 2
    blank_image = np.ones((int(image_size[1]), int(image_size[0]), 3), np.uint8) * 255

    # draw scene lines
    draw_lines(blank_image, scene_lines, (0, 0, 255))

    # draw model lines
    if meta["ransac_type"] == "opencv":
        model_lines_transformed = transform_line_batch(model_lines_postprocess,
                                                       np.array(transformation_2d[:2, :2]),
                                                       np.array(transformation_2d[:, 2]),
                                                       center)
    elif meta["ransac_type"] == "final" or meta["ransac_type"] == "pairwise":
        r = get_rotation_matrix_2d(transformation_2d[1])
        model_lines_transformed = transform_line_batch(model_lines_postprocess,
                                                       r,
                                                       np.array(transformation_2d[2:3]),
                                                       center)

    else:
        model_lines_transformed = transform_line_batch(model_lines_postprocess,
                                                       transformation_2d[0],
                                                       transformation_2d[1:3],
                                                       center)

    draw_lines(blank_image, model_lines_transformed, (0, 255, 0))

    # draw the connection lines
    connection_lines = []
    for match in matching_correspondence:
        p1 = get_midpoint(model_lines_transformed[int(match[0])])
        p2 = get_midpoint(scene_lines_postprocess[int(match[1])])
        connection_lines += [[float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])]]

    draw_lines(blank_image, connection_lines, (0, 0, 0))

    # write image to disc
    save_image(blank_image, meta["path"] + dump_path + "matching_visuals" + str(meta["test_set_number"]) + ".png")

    # show to screen, block for user input
    if not meta["system"] == "Jetson Board":
         plot_image(blank_image, "test set " + str(meta["test_set_number"]), True)



def get_midpoint(line):

    p1 = line[0:2]
    p2 = line[2:4]

    return ((p2 - p1) / 2) + p1


if __name__ == "__main__":

    # options: "standard" "cuda" "first" "simple" "pairwise" "final"
    algorithm_list = ["final"]

    # options: 1 - 70
    test_list = [2, 50, 5, 10, 12, 22, 24, 25, 37, 43, 51, 53, 62, 67, 69]

    for test_num in test_list:
        for algo in algorithm_list:

            # this is for measurement repeatability
            for i in range(1):

                # change seed for different random results
                seed = 2000 + i
                do_test_run(test_num, algo, seed)
