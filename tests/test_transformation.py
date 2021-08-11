from modules.components.batch_transformation import transform_line_batch, get_rotation_matrix_2d
from modules.components.transformation import define_transformation_opencv, define_transformation_numpy, \
    define_transformation_numpy_pairs, define_transformation_pair_opencv, define_transformation_pair_midpoint_opencv, \
    define_transformation, define_perspective_transformation
from modules.handlers.load_test_sets import load_test_set
from sys import platform
import numpy as np
import time
import timeit

def load_data(set_num):

    path = "../"

    if platform == "linux" or platform == "linux2":
        path = "/home/lukas/jetson-cuda/"

    scene_lines, model_lines, match_id_list = load_test_set(set_num, path)


    # get rid of unnecessary data
    #scene_lines = [[p1[0][0], p1[1][0], p2[0][0], p2[1][0], mid] for p1, p2, vec, mid, len, ang in scene_lines]
    #model_lines = [[p1[0][0], p1[1][0], p2[0][0], p2[1][0], mid] for p1, p2, vec, _, len, ang, _, mid, _, _, _ in
    #               model_lines]

    # give ids to lines
    scene_lines = [[line[0], line[1], line[2], line[3], num] for num, line in enumerate(scene_lines)]
    model_lines = [[line[0], line[1], line[2], line[3], num] for num, line in enumerate(model_lines)]

    return scene_lines, model_lines


def test_transformation():

    scene_lines, model_lines = load_data(2)
    model_lines_transformed = transform_line_batch(model_lines,
                                                   get_rotation_matrix_2d(0),
                                                   np.array([25,0]),
                                                   np.array([256, 256]))

    start_time = time.time()
    m1 = define_transformation_pair_midpoint_opencv(model_lines[1], model_lines[2],
                                                    model_lines_transformed[1], model_lines_transformed[2])
    duration = (time.time() - start_time)
    print("OPENCV midpoint affine transformation")
    print(m1)
    print("took " + str(duration) + " seconds")
    print()


    start_time = time.time()
    m4 = define_transformation_pair_opencv(model_lines[1], model_lines[2],
                                                    model_lines_transformed[1], model_lines_transformed[2])
    duration = (time.time() - start_time)
    print("OPENCV selective affine transformation")
    print(m4)
    print("took " + str(duration) + " seconds")
    print()


    start_time = time.time()
    m2 = define_transformation_numpy_pairs(model_lines[1], model_lines[2],
                                                    model_lines_transformed[1], model_lines_transformed[2])
    duration = (time.time() - start_time)
    print("NUMPY midpoint affine transformation")
    print(m2)
    print("took " + str(duration) + " seconds")
    print()

    start_time = time.time()
    m6 = define_transformation_numpy_pairs(model_lines[1], model_lines[2],
                                                    model_lines_transformed[1], model_lines_transformed[2])
    duration = (time.time() - start_time)
    print("NUMPY selective affine transformation")
    print(m6)
    print("took " + str(duration) + " seconds")
    print()


    start_time = time.time()
    m3 = define_transformation(np.array((model_lines[1], model_lines[2])),
                                np.array((model_lines_transformed[1], model_lines_transformed[2])),
                               np.array([256, 256]))
    duration = (time.time() - start_time)
    print("SELF rigid transformation")
    print(m3)
    print("took " + str(duration) + " seconds")
    print()




    start_time = time.time()
    m5 = define_perspective_transformation(model_lines[1], model_lines[2],
                                                    model_lines_transformed[1], model_lines_transformed[2])
    duration = (time.time() - start_time)
    print("OPENCV perspective transformation")
    print(m5)
    print("took " + str(duration) + " seconds")
    print()



    ###################

    setup_code = """from modules.components.transformation import define_transformation
import numpy as np
from modules.handlers.load_test_sets import load_test_set
from modules.components.batch_transformation import transform_line_batch, get_rotation_matrix_2d
from sys import platform

def load_data(set_num):

    path = "../"

    if platform == "linux" or platform == "linux2":
        path = "/home/lukas/jetson-cuda/"

    scene_lines, model_lines, match_id_list = load_test_set(set_num, path)


    # get rid of unnecessary data
    #scene_lines = [[p1[0][0], p1[1][0], p2[0][0], p2[1][0], mid] for p1, p2, vec, mid, len, ang in scene_lines]
    #model_lines = [[p1[0][0], p1[1][0], p2[0][0], p2[1][0], mid] for p1, p2, vec, _, len, ang, _, mid, _, _, _ in
    #               model_lines]

    # give ids to lines
    scene_lines = [[line[0], line[1], line[2], line[3], num] for num, line in enumerate(scene_lines)]
    model_lines = [[line[0], line[1], line[2], line[3], num] for num, line in enumerate(model_lines)]

    return scene_lines, model_lines
    
scene_lines, model_lines = load_data(2)
model_lines_transformed = transform_line_batch(model_lines,
                                               get_rotation_matrix_2d(0),
                                               np.array([25,0]),
                                               np.array([256, 256]))
"""

    main_code = """m3 = define_transformation(np.array((model_lines[1], model_lines[2])),
                                np.array((model_lines_transformed[1], model_lines_transformed[2])),
                               np.array([256, 256]))
    """

    number_revs = 1000
    time_taken =  timeit.timeit(stmt=main_code,
                    setup=setup_code,
                    number=number_revs)
    time_avg = time_taken / number_revs

    print("average time: " + str(time_avg))



if __name__ == "__main__":

    test_transformation()