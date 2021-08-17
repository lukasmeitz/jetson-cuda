import time
from concurrent.futures import thread
from sys import platform
from threading import Thread, Event, Lock

from modules.algorithms.preprocessing import filter_lines
from modules.algorithms.ransac_stage_1 import ransac_stage_1
from modules.components.preprocessor import preprocess_length
from modules.handlers.load_test_sets import load_test_set, create_line_permutations
from modules.optimized.parallel_threaded_standard import ransac_parallel_threaded
from modules.visuals.imaging import draw_lines, plot_image, load_image

import numpy as np


def draw_current(image, model_lines):
    draw_lines(image, model_lines, color=(0, 0, 255))
    plot_image(image, "Current Iteration", blocking=False)


lock = Event()
sync = Lock()


def observer_thread(to_observe):

    while True:
        sync.acquire()
        #print(to_observe[-1])

        if lock.is_set():
            break


def drawing_thread(model_lines_transformed, scene_lines, img):

    while True:
        sync.acquire()
        if model_lines_transformed[0] is not 0:

            image = img[0].copy()
            draw_current(image, scene_lines[0])
            #draw_current(image, model_lines_transformed[0])

        if lock.is_set():
            break


def run_demonstration(path):

    test_set_ids = [i for i in range(1, 21)]

    seed = 2021
    rng = np.random.default_rng(seed)

    result_transformation = [0]
    model_lines_transformed = [0]
    scene_lines, model_lines = [0], [0]
    image = [0]


    # start observing
    t = Thread(target=observer_thread, args=(result_transformation, ))
    t.start()

    t2 = Thread(target=drawing_thread, args=(model_lines_transformed, scene_lines, image))
    t2.start()


    for set_id in test_set_ids:

        print("starting set #" + str(set_id))

        if set_id >= 50:
            center_point = np.array([1280/2, 720/2])
        else:
            center_point = np.array([256, 256])

        # load data
        scene_lines[0], model_lines[0], match_id_list = load_test_set(set_id, path, demo=True)
        set_num = "{:03d}".format(set_id)
        image[0] = load_image(path + 'data/TestSets_Demonstrator/TestSet' + set_num + '/image.png')

        # preprocess
        #scene_lines[0] = filter_lines(scene_lines[0], max_lines=120)
        #model_lines[0] = filter_lines(model_lines[0], max_lines=120)

        scene_lines[0] = np.array(scene_lines[0])
        model_lines[0] = np.array(model_lines[0])

        scene_lines[0] = preprocess_length(scene_lines[0], max_lines=150)
        model_lines[0] = preprocess_length(model_lines[0], max_lines=150)

        scene_lines[0] = np.array(scene_lines[0])
        model_lines[0] = np.array(model_lines[0])

        #print(scene_lines[0])

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
    t.join()
    t2.join()
    print("finished")


if __name__ == "__main__":

    res_path = ""

    if platform == "linux" or platform == "linux2":
        res_path = "/home/lukas/jetson-cuda/"

    run_demonstration(res_path)
