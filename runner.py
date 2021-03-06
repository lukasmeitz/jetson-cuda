import datetime
import json
import os
import time
from sys import platform

from modules.algorithms.adaptive_ransac import adaptive_ransac
from modules.algorithms.preprocessing import filter_lines
from modules.algorithms.randomised_ransac import randomised_ransac
from modules.algorithms.ransac_draft import ransac_draft
from modules.handlers import *
from modules.handlers.load_test_sets import load_test_set, create_line_permutations
from modules.optimized.optimized_math import transform_line_batch
from modules.optimized.optimized_ransac import optimized_ransac
from modules.visuals.imaging import *
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

    seed = 2021
    rng = np.random.default_rng(seed)

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

    # shuffle lines
    np.random.shuffle(scene_lines)
    np.random.shuffle(model_lines)


    '''
    preprocessing
    '''

    # transformation onto the scenelines
    scene_lines = transform_line_batch(scene_lines, 0.0, np.array([0, 0]), np.array([256, 256]))

    # start preprocessing logic
    scene_lines = filter_lines(scene_lines)
    model_lines = filter_lines(model_lines)

    # create scene line permutations
    scene_permutations = create_line_permutations(scene_lines)
    model_permutations = create_line_permutations(model_lines)

    # add permuation count
    meta["scene_line_pairs"] = len(scene_permutations)
    meta["model_line_pairs"] = len(model_permutations)


    '''
    algorithm
    '''

    # feed this to ransac
    if meta["ransac_type"] == "standard":
        start_time = time.time()
        matching_correspondence, transformation_2d = ransac_draft(model_permutations, scene_permutations, rng)
        meta["duration"] = (time.time() - start_time)

    elif meta["ransac_type"] == "randomised":
        start_time = time.time()
        matching_correspondence, transformation_2d = randomised_ransac(model_permutations, scene_permutations, rng)
        meta["duration"] = (time.time() - start_time)

    elif meta["ransac_type"] == "adaptive":
        start_time = time.time()
        matching_correspondence, transformation_2d = adaptive_ransac(model_permutations, scene_permutations, rng)
        meta["duration"] = (time.time() - start_time)

    elif meta["ransac_type"] == "optimized":
        start_time = time.time()
        matching_correspondence, transformation_2d = optimized_ransac(model_permutations, scene_permutations, rng)
        meta["duration"] = (time.time() - start_time)


    '''
    post processing
    '''

    # data structure for line ids: [(id_model_1, id_scene_1), ..., (id_model_n, id_scene_m)]
    matched_lines = []

    # get line ids from pair ids
    for pair in matching_correspondence:

        # model line and scene line ids
        ml = pair[0]
        sl = pair[1]

        # model line ids
        ml1_id = int(model_permutations[ml][0][4])
        ml2_id = int(model_permutations[ml][1][4])

        # scene line ids
        sl1_id = int(scene_permutations[sl][0][4])
        sl2_id = int(scene_permutations[sl][1][4])

        # save the matching relation
        if (ml1_id, sl1_id) not in matched_lines:
            matched_lines += [(ml1_id, sl1_id)]

        if (ml2_id, sl2_id) not in matched_lines:
            matched_lines += [(ml2_id, sl2_id)]

    meta["match_id_list"] = matched_lines
    meta["match_count"] = len(matched_lines)
    meta["transformation"] = transformation_2d

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
    draw_lines(blank_image, model_lines, (0, 255, 0))

    # draw the connection lines
    connection_lines = []
    for match in matched_lines:
        p1 = model_lines_postprocess[match[0]][4]
        p2 = scene_lines_postprocess[match[1]][4]
        connection_lines += [[float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])]]

    draw_lines(blank_image, connection_lines, (255, 255, 0))

    # write image to disc
    save_image(blank_image, meta["path"] + dump_path + "matching_visuals" + str(meta["test_set_number"]) + ".png")

    # show to screen, block for user input
    # if not meta["system"] == "Jetson Board":
    #     plot_image(blank_image, "test set " + str(meta["test_set_number"]), True)


if __name__ == "__main__":

    algorithm_list = ["adaptive"]  # ["optimized", "adaptive", "standard", "randomised"]

    for test_num in range(40, 41, 1):

        for algo in algorithm_list:
            do_test_run(test_num, algo)
