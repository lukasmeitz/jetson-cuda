import itertools
import json
import time
from sys import platform

from modules.algorithms.randomised_ransac import randomised_ransac
from modules.algorithms.ransac_draft import ransac_draft
from modules.handlers import *
from modules.handlers.load_test_sets import load_test_set
from modules.line_matcher.define_transform import transform_modelline_batch, transform_line_batch
from modules.visuals.imaging import *
from tests.test_imaging import *
from tests.test_gtm_handler import *

if __name__ == "__main__":

    '''
    parameters
    '''
    # dict to gather test run information
    meta = {}

    # select a single line matching test set between 1 and 45
    meta["test_set_number"] = 8
    meta["ransac_type"] = "randomised"

    # determine platform and configure resource path
    if platform == "linux" or platform == "linux2":
        meta["path"] = "/home/lukas/jetson-cuda/"
        meta["system"] = "Jetson Board"
    else:
        meta["path"] = ""
        meta["system"] = "Windows Desktop"

    '''
    data loading
    '''
    scene_lines, model_lines, match_id_list = load_test_set(meta["test_set_number"], meta["path"])
    meta["scene_lines"] = len(scene_lines)
    meta["model_lines"] = len(model_lines)
    meta["match_id_list_gtm"] = [int(mid)-1 for mid in match_id_list if mid != 0]
    meta["match_count_gtm"] = len(meta["match_id_list_gtm"])

    # get rid of unnecessary data
    scene_lines = [[p1[0][0], p1[1][0], p2[0][0], p2[1][0], mid] for p1, p2, vec, mid, len, ang in scene_lines]
    model_lines = [[p1[0][0], p1[1][0], p2[0][0], p2[1][0], mid] for p1, p2, vec, _, len, ang, _, mid, _, _, _ in model_lines]

    scene_lines_postprocess = scene_lines
    model_lines_postprocess = model_lines

    # give ids to lines
    scene_lines = [[line[0], line[1], line[2], line[3], num] for num, line in enumerate(scene_lines)]
    model_lines = [[line[0], line[1], line[2], line[3], num] for num, line in enumerate(model_lines)]

    # shuffle lines
    np.random.shuffle(scene_lines)
    np.random.shuffle(model_lines)

    '''
    preprocessing
    '''

    # transformation onto the scenelines
    scene_lines = transform_line_batch(scene_lines, 0.0, [0, 0], [256, 256])

    # create scene line permutations
    scene_permutations = [[l1, l2] for l1, l2 in list(itertools.combinations(scene_lines, r=2))]
    model_permutations = [[l1, l2] for l1, l2 in list(itertools.combinations(model_lines, r=2))]

    meta["scene_line_pairs"] = len(scene_permutations)
    meta["model_line_pairs"] = len(model_permutations)

    '''
    algorithm
    '''

    # start timer
    start_time = time.time()

    # feed this to ransac

    if meta["ransac_type"] == "standard":
        matching_correspondence, transformation_2d = ransac_draft(model_permutations, scene_permutations)

    elif meta["ransac_type"] == "randomised":
        matching_correspondence, transformation_2d = randomised_ransac(model_permutations, scene_permutations)

    # stop time
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
        ml1_id = model_permutations[ml][0][4]
        ml2_id = model_permutations[ml][1][4]

        # scene line ids
        sl1_id = scene_permutations[sl][0][4]
        sl2_id = scene_permutations[sl][1][4]

        # save the matching relation
        if (ml1_id, sl1_id) not in matched_lines:
            matched_lines += [(ml1_id, sl1_id)]

        if (ml2_id, sl2_id) not in matched_lines:
            matched_lines += [(ml2_id, sl2_id)]

    meta["match_id_list"] = matched_lines
    meta["match_count"] = len(matched_lines)
    meta["transformation"] = transformation_2d

    print(meta)
    with open(meta["path"] + "results/matching_testset" + str(meta["test_set_number"]) + ".json", 'w') as fp:
        json.dump(meta, fp)


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
    save_image(blank_image, meta["path"] + "results/matching_testset" + str(meta["test_set_number"]) + ".png")

    # show to screen, block for user input
    if not meta["system"] == "Jetson Board":
        plot_image(blank_image, "test set " + str(meta["test_set_number"]), True)
