
import math
import numpy as np

from numba import cuda

from modules.components.evaluation import calc_min_distance
from modules.components.preprocessor import preprocessor
from modules.components.transformation import define_transformation
from modules.handlers.load_test_sets import load_test_set


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



def ransac_cuda_optimized(model_lines, scene_lines,
                              model_line_indices, scene_line_indices,
                              random_generator, center):

    # debug info
    print("max combinations for evaluation: " + str(len(model_lines) * len(scene_lines)))

    # Parameters
    iterations = 50
    threshold = 25
    max_inliers = -1
    best_transformation = []

    # GPU setup
    threadsperblock = (16, 16)
    blockspergrid_x = np.math.ceil(model_lines.shape[0] / threadsperblock[0])
    blockspergrid_y = np.math.ceil(model_lines.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # generate samples
    random_sample_indices = random_generator.random((iterations, 2))
    random_sample_indices *= [len(model_line_indices)-1, len(scene_line_indices)-1]
    random_sample_indices = np.round(random_sample_indices).astype(int)

    # loop through
    for i in range(iterations):

        # resolve index
        # print("picking " + str(random_sample_indices[i]))
        model_pair_index = model_line_indices[random_sample_indices[i][0]]
        scene_pair_index = scene_line_indices[random_sample_indices[i][1]]

        # define transform
        transformation = define_transformation(
            np.array([model_lines[int(model_pair_index[0])],
                      model_lines[int(model_pair_index[1])]]),
            np.array([scene_lines[int(scene_pair_index[0])],
                      scene_lines[int(scene_pair_index[1])]]),
            center)


        # batch transformation
        model_lines_transformed = np.copy(model_lines)
        transform_line_batch[blockspergrid, threadsperblock](model_lines_transformed,
                                                             np.array(transformation[0]),
                                                             np.array(transformation[1:3]),
                                                             center)

        # evaluation
        results = np.zeros((len(model_lines_transformed) * len(scene_lines)))
        evaluate_line_batch[blockspergrid, threadsperblock](model_lines_transformed, scene_lines, results)

        matches = []
        inliers = 0
        for r in results:
            if 0.0 < r < threshold:
                inliers += 1

        print("found " + str(inliers) + " inliers")

        if inliers > max_inliers:
            max_inliers = inliers
            best_transformation = transformation
            # print(matches)

    # return
    # print("found max " + str(max_inliers) + " inliers")
    return matches, best_transformation


if __name__ == "__main__":

    # prepare
    center =  np.array([1280/2, 720/2])  # np.array([256, 256])
    set = 70
    seed = 2000
    rng = np.random.default_rng(seed)

    # preprocess
    scene_lines, model_lines, match_id_list = load_test_set(set, "../../")
    model_lines = np.array(model_lines)
    scene_lines = np.array(scene_lines)
    scene_line_pairs = preprocessor(scene_lines, max_lines=120)
    model_line_pairs = preprocessor(model_lines, max_lines=120)

    # sample
    matches, transform = ransac_cuda_optimized(model_lines, scene_lines,
                              model_line_pairs, scene_line_pairs,
                              rng, center)
    print(matches)
