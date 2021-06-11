import time
from concurrent.futures import thread
from sys import platform
from threading import Thread, Event, Lock

from modules.algorithms.preprocessing import filter_lines
from modules.algorithms.ransac_stage_1 import ransac_stage_1
from modules.handlers.load_test_sets import load_test_set, create_line_permutations
from modules.visuals.imaging import draw_lines, plot_image

import numpy as np


def draw_current(image, model_lines):
    draw_lines(image, model_lines, color=(0, 200, 0))
    plot_image(image, "Current Iteration", blocking=False)


lock = Event()
sync = Lock()


def observer_thread(to_observe):

    while True:
        sync.acquire()
        print(to_observe[-1])

        if lock.is_set():
            break


def drawing_thread(model_lines_transformed, scene_lines):

    while True:
        sync.acquire()
        if model_lines_transformed[0] is not 0:
            image = np.ones((512, 512, 3), np.uint8) * 255
            draw_lines(image, scene_lines[0], color=(0, 0, 255))
            draw_current(image, model_lines_transformed[0])

        if lock.is_set():
            break


def run_demonstration(path):

    test_set_ids = [i for i in range(1, 6)]
    center_point = np.array([256, 256])

    seed = 2021
    rng = np.random.default_rng(seed)

    result_transformation = [0]
    result_inliers = [0]
    model_lines_transformed = [0]
    scene_lines, model_lines = [0], [0]


    # start observing
    t = Thread(target=observer_thread, args=(result_transformation, ))
    #t.start()

    t2 = Thread(target=drawing_thread, args=(model_lines_transformed, scene_lines))
    t2.start()


    for set_id in test_set_ids:

        # load data
        scene_lines[0], model_lines[0], match_id_list = load_test_set(set_id, path)

        # preprocess
        scene_lines[0] = filter_lines(scene_lines[0])
        model_lines[0] = filter_lines(model_lines[0])

        # match using ransac
        ransac_stage_1(model_lines[0],
                       scene_lines[0],
                       center_point,
                       rng,
                       result_transformation,
                       model_lines_transformed,
                       sync)

    lock.set()
    if sync.locked():
        sync.release()
    #t.join()
    t2.join()
    print("finished")


if __name__ == "__main__":

    res_path = ""

    if platform == "linux" or platform == "linux2":
        res_path = "/home/lukas/jetson-cuda/"

    run_demonstration(res_path)
