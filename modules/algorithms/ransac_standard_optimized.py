from modules.components.batch_transformation import transform_line_batch
from modules.components.evaluation import calc_min_distance, calc_min_pair_distance
from modules.components.preprocessor import preprocessor
from modules.components.transformation import define_transformation_pair_midpoint_opencv, \
    define_transformation_pair_opencv, define_transformation
from modules.handlers.load_test_sets import load_test_set

import numpy as np


def ransac_standard_optimized(model_lines, scene_lines,
                              model_line_indices, scene_line_indices,
                              random_generator, center):

    # Parameters
    iterations = 50
    threshold = 25
    max_inliers = -1
    best_transformation = []
    print(str(len(model_line_indices)) + " model line pairs")
    print(str(len(scene_line_indices)) + " scene line pairs")

    # generate samples
    random_sample_indices = random_generator.random((iterations, 2))
    random_sample_indices *= [len(model_line_indices)-1, len(scene_line_indices)-1]
    random_sample_indices = np.round(random_sample_indices).astype(int)
    print("max index model: " + str(max(random_sample_indices[:, 0])))
    print("max index scene: " + str(max(random_sample_indices[:, 1])))

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
        # print(transformation)
        # print(np.transpose(transformation[:, 2]))

        # batch transformation
        model_lines_transformed = transform_line_batch(model_lines,
                                                            transformation[0],
                                                            transformation[1:3],
                                                            center)
        # print("transformed " + str(len(model_lines_transformed)) + " lines")

        # evaluation
        inliers = 0
        inlier_features = 0
        matches = []
        for m in range(len(model_lines_transformed)):
            for s in range(len(scene_lines)):

                if calc_min_distance(model_lines_transformed[m], scene_lines[s]) < threshold:
                    inliers += 1
                    matches.append([model_lines_transformed[m][6], scene_lines[s][6]])

        if inliers > max_inliers:
            max_inliers = inliers
            best_transformation = transformation
            print("found " + str(inliers) + " inliers")
            print(matches)

    # return
    print("found max " + str(max_inliers) + " inliers")
    return matches, best_transformation


if __name__ == "__main__":

    # prepare
    center =  np.array([1280/2, 720/2])  # np.array([256, 256])
    set = 70
    seed = 2000
    rng = np.random.default_rng(seed)

    # preprocess
    scene_lines, model_lines, match_id_list = load_test_set(set, "../../")
    scene_line_pairs = preprocessor(scene_lines, max_lines=120)
    model_line_pairs = preprocessor(model_lines, max_lines=120)
    model_lines = np.array(model_lines)
    scene_lines = np.array(scene_lines)

    # sample
    ransac_standard_optimized(model_lines, scene_lines,
                              model_line_pairs, scene_line_pairs,
                              rng, center)
