import math

from numba import cuda
import numpy as np

from modules.algorithms.preprocessing import filter_lines
from modules.handlers.load_test_sets import load_test_set
from modules.optimized.optimized_math import calc_min_line_distance



@cuda.jit
def add_to_a_2D_array(an_array, transform):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < 4:

        # transform point
       an_array[x][y] += transform[y]




@cuda.jit
def iterate_2D_array(an_array, result):

    result = 999.0
    i = cuda.grid(1)

    if i < an_array.shape[0]:
        if result < an_array[i]:
            result = an_array[i]



@cuda.jit
def evaluate_line_batch(lines, reference_lines, scores):

    m, s = cuda.grid(2)
    if m < lines.shape[0] and s < reference_lines.shape[0]:

        # cross distances
        scores[((lines.shape[0]-1) * s) + m] = math.sqrt((lines[m, 0] - reference_lines[s, 0])**2)

        scores[((lines.shape[0]-1) * s) + m] = max(
            scores[((lines.shape[0] - 1) * s) + m],
            math.sqrt((lines[m, 1] - reference_lines[s, 1])**2)
        )

        scores[((lines.shape[0]-1) * s) + m] = max(
            scores[((lines.shape[0] - 1) * s) + m],
            math.sqrt((lines[m, 2] - reference_lines[s, 2])**2)
        )

        scores[((lines.shape[0]-1) * s) + m] = max(
            scores[((lines.shape[0] - 1) * s) + m],
            math.sqrt((lines[m, 3] - reference_lines[s, 3])**2)
        )

        # accumulate
        #scores[((lines.shape[0]-1) * s) + m] = (lines[m, 0] - reference_lines[s, 0])
        #scores[(lines.shape[0] * s) + m] += (lines[m, 1] - reference_lines[s, 1])
        #scores[(lines.shape[0] * s) + m] += (lines[m, 2] - reference_lines[s, 2])
        #scores[(lines.shape[0] * s) + m] += (lines[m, 3] - reference_lines[s, 3])

        # reduce
        scores[(lines.shape[0] * s) + m] /= 4



@cuda.jit
def transform_line_batch(lines, rotation, transformation_distance, center_point):

    x = cuda.grid(1)
    if x < lines.shape[0]:


        # translate to origin
        lines[x, 0] -= center_point[0]
        lines[x, 1] -= center_point[1]
        # rotate
        lines[x, 0] = lines[x, 0] * rotation[0, 0] + lines[x, 1] * rotation[0, 1]
        lines[x, 1] = lines[x, 0] * rotation[1, 0] + lines[x, 1] * rotation[1, 1]
        # translate forth
        lines[x, 0] += center_point[0] + transformation_distance[0]
        lines[x, 1] += center_point[1] + transformation_distance[1]


        # p2
        lines[x, 2] -= center_point[0]
        lines[x, 3] -= center_point[1]
        # rotate
        lines[x, 2] = lines[x, 2] * rotation[0, 0] + lines[x, 3] * rotation[0, 1]
        lines[x, 3] = lines[x, 2] * rotation[1, 0] + lines[x, 3] * rotation[1, 1]
        # translate forth
        lines[x, 2] += center_point[0] + transformation_distance[0]
        lines[x, 3] += center_point[1] + transformation_distance[1]



if __name__ == "__main__":

    # input
    scene_lines, model_lines, match_id_list = load_test_set(10, '../../')
    arr = np.ones((200, 6))
    print(arr.dtype)
    arr_ref = np.ones((200, 6)) * 3.0

    # output
    arr_results = np.zeros((len(model_lines) * len(scene_lines)))
    print("len res array: " + str(len(arr_results)))

    # parameter
    tr = np.array([5.0, 1.0])
    center = np.array([5.0, 5.0])
    rotation_angle = 0.0

    # rotation matrix
    r = np.array(((np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))),
                  (np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle)))))



    # preprocess
    scene_lines = filter_lines(scene_lines, max_lines=80)
    model_lines = filter_lines(model_lines, max_lines=50)
    print(model_lines)
    model_lines = np.array(model_lines).astype('float64')
    print(model_lines.dtype)



    # GPU setup
    threadsperblock = (16, 16)
    blockspergrid_x = np.math.ceil(arr.shape[0] / threadsperblock[0])
    blockspergrid_y = np.math.ceil(arr.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)




    # test array transform
    result = np.ones((100,4))
    add_to_a_2D_array[blockspergrid, threadsperblock](result, np.array([2.0, 2.0, 2.0, 2.0]))
    print(result)

    # transform batch invocation
    transform_line_batch[blockspergrid, threadsperblock](arr, r, tr, center)


    # evaluation invocation
    #evaluate_line_batch[blockspergrid, threadsperblock](model_lines, scene_lines, arr_results)

    # print scores
    print(arr_results)

    # find smallest error
    #iterate_2D_array[blockspergrid, threadsperblock](arr_results, result)

    result = 9999.9
    for n, r in enumerate(arr_results):
        if result > r > 0.0:
            result = r
            idx = n

    # count inlier
    inlier = 0
    threshold = 25
    for r in arr_results:
        if r < threshold:
            inlier += 1


    print(inlier)
