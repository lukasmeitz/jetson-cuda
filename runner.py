import datetime
import json
import os
import time
from sys import platform

from modules.algorithms.ransac_cuda_optimized import ransac_cuda_optimized
from modules.algorithms.ransac_standard_optimized import ransac_standard_optimized
from modules.components.batch_transformation import transform_line_batch
from modules.components.preprocessor import preprocessor
from modules.handlers.load_test_sets import load_test_set
from tests.test_imaging import *
from tests.test_gtm_handler import *


def do_test_run(set_number, algorithm):

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
    dump_path = "results/" + str(algorithm) + "/Set_" + "{:03d}".format(set_number) + "/" + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) + "/"

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    else:
        print("skipping")
        return

    seed = 3268
    rng = np.random.default_rng(seed)
    center = np.array([1280/2, 720/2]) if set_number < 51 else np.array([256, 256])

    '''
    data loading
    '''

    scene_lines, model_lines, match_id_list = load_test_set(meta["test_set_number"], meta["path"])
    meta["scene_lines"] = len(scene_lines)
    meta["model_lines"] = len(model_lines)
    meta["match_id_list_gtm"] = [int(mid)-1 for mid in match_id_list if mid != 0]
    meta["match_count_gtm"] = len(meta["match_id_list_gtm"])

    scene_lines_postprocess = scene_lines
    model_lines_postprocess = model_lines


    '''
    preprocessing
    '''

    # transformation onto the scenelines
    #scene_lines = transform_line_batch(scene_lines, 0.0, np.array([0, 0]), center_point=center)

    # start preprocessing logic
    scene_line_pairs = preprocessor(scene_lines)
    model_line_pairs = preprocessor(model_lines)

    # add permuation count
    meta["scene_line_pairs"] = len(scene_line_pairs)
    meta["model_line_pairs"] = len(model_line_pairs)


    '''
    algorithm
    '''

    # feed this to ransac
    if meta["ransac_type"] == "standard":
        start_time = time.time()
        matching_correspondence, transformation_2d = ransac_standard_optimized(model_lines, scene_lines,
                                                                               model_line_pairs, scene_line_pairs,
                                                                               rng, center)
        meta["duration"] = (time.time() - start_time)

    elif meta["ransac_type"] == "cuda":
        start_time = time.time()
        matching_correspondence, transformation_2d = ransac_cuda_optimized(model_lines, scene_lines,
                                                                               model_line_pairs, scene_line_pairs,
                                                                               rng, center)
        meta["duration"] = (time.time() - start_time)


    '''
    post processing
    '''

    meta["match_id_list"] = matching_correspondence
    meta["match_count"] = len(matching_correspondence)

    #print(transformation_2d)
    #meta["transformation"] = transformation_2d

    with open(meta["path"] + dump_path + "matching_result" + str(meta["test_set_number"]) + ".json", 'w') as fp:
        json.dump(meta, fp)

    print("found " + str(meta["match_count"]) + " of " + str(meta["match_count_gtm"]) + " matches")
    print()

    '''
    visualization
    '''

    blank_image = np.ones((512,512, 3), np.uint8) * 255

    # draw scene lines
    draw_lines(blank_image, scene_lines, (0, 0, 255))

    # draw model lines
    model_lines_transformed = transform_line_batch(model_lines,
                                                   transformation_2d[0],
                                                   transformation_2d[1:3],
                                                   center)
    draw_lines(blank_image, model_lines_transformed, (0, 255, 0))

    # draw the connection lines
    connection_lines = []
    #for match in matching_correspondence:
        #p1 = model_lines_postprocess[int(match[0])][7]
        #p2 = scene_lines_postprocess[int(match[1])][7]
        #connection_lines += [[float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])]]

    #draw_lines(blank_image, connection_lines, (255, 255, 0))

    # write image to disc
    save_image(blank_image, meta["path"] + dump_path + "matching_visuals" + str(meta["test_set_number"]) + ".png")

    # show to screen, block for user input
    if not meta["system"] == "Jetson Board":
         plot_image(blank_image, "test set " + str(meta["test_set_number"]), True)


if __name__ == "__main__":

    algorithm_list = ["cuda", "standard"]
    test_list = [5, 10, 65, 66, 67, 68, 69, 70]

    for test_num in test_list:

        for algo in algorithm_list:
            do_test_run(test_num, algo)
