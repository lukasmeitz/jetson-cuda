import time

import numpy as np

from modules.algorithms.ransac_cuda_final import get_transformation_cuda
from modules.components.transformation import define_transformation
from modules.handlers.load_test_sets import load_test_set


def ransac_split_evaluation(model_lines,
                      scene_lines,
                      random_generator,
                      center,
                      threshold=40,
                      iterations=500):

    # generate indices
    indices = random_generator.random((iterations, 4))
    indices *= [len(model_lines) - 1,
                len(model_lines) - 1,
                len(scene_lines) - 1,
                len(scene_lines) - 1]
    indices = np.round(indices).astype(int)

    # parallel sampling
    start_parallel_sampling = time.time()
    transformations = np.zeros((len(indices), 4))
    threadsperblock = (16, 16)
    blockspergrid_x = np.math.ceil(len(indices) / threadsperblock[0])
    blockspergrid_y = np.math.ceil(len(scene_lines) / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    get_transformation_cuda[blockspergrid, threadsperblock](model_lines, scene_lines, indices, transformations)

    time_parallel_sampling = time.time() - start_parallel_sampling
    print("cuda took " + str(time_parallel_sampling))

    # normal sampling
    start_standard_sampling = time.time()
    for i in range(iterations):

        # define transform
        transformation = define_transformation(
            np.array([model_lines[int(indices[i][0])],
                      model_lines[int(indices[i][1])]]),
            np.array([scene_lines[int(indices[i][2])],
                      scene_lines[int(indices[i][3])]]),
            center)

        # bail-out-test
        w = np.rad2deg(np.arccos(transformation[0][0, 0]))
        t = np.sum(np.abs(transformation[1:3]))
        if t > 100 or w > 60:
            continue
    time_standard_sampling = time.time() - start_standard_sampling
    print("standard took " + str(time_standard_sampling))


if __name__ == "__main__":

    # prepare
    center =  np.array([1280/2, 720/2])  # np.array([256, 256])
    set = 70
    seed = 2000
    rng = np.random.default_rng(seed)

    # preprocess
    scene_lines, model_lines, match_id_list = load_test_set(set, "../")
    model_lines = np.array(model_lines)
    scene_lines = np.array(scene_lines)

    # sample
    ransac_split_evaluation(model_lines,
                            scene_lines,
                            rng,
                            center)
