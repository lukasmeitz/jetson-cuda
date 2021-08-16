import math
import numpy as np

from modules.components.transformation import define_transformation
from modules.handlers.load_test_sets import load_test_set
from modules.line_matcher.define_transform import transform_line_batch


def get_transformation_rigid(model_lines, scene_lines, indices, center, transformations):

    for x in range(len(indices)):

        # line layout: line = [p1x, p1y, p2x, p2y]

        # first and second model line
        ml1 = model_lines[indices[x][0]]
        ml2 = model_lines[indices[x][1]]

        # first and second scene line
        sl1 = scene_lines[indices[x][2]]
        sl2 = scene_lines[indices[x][3]]

        # calculate angle between lines 1
        length_ml1 = math.sqrt(((ml1[3] - ml1[1]) ** 2) + ((ml1[2] - ml1[0]) ** 2))
        ml1x_norm = (ml1[2] - ml1[0]) / length_ml1
        ml1y_norm = (ml1[3] - ml1[1]) / length_ml1

        length_sl1 = math.sqrt(((sl1[2] - sl1[0]) ** 2) + ((sl1[3] - sl1[1]) ** 2))
        sl1x_norm = (sl1[2] - sl1[0]) / length_sl1
        sl1y_norm = (sl1[3] - sl1[1]) / length_sl1

        cos_1 = ml1x_norm * sl1x_norm + ml1y_norm * sl1y_norm
        cos_2 = -ml1y_norm * sl1x_norm + ml1x_norm * sl1y_norm
        winkel_radians = math.acos(min(math.fabs(cos_1), 1))
        w1 = np.degrees(winkel_radians)

        if cos_1 * cos_2 < 0:
            w1 = -w1

        # calculate angle between lines 2
        length_ml2 = math.sqrt(((ml2[2] - ml2[0]) ** 2) + ((ml2[3] - ml2[1]) ** 2))
        ml2x_norm = (ml2[2] - ml2[0]) / length_ml2
        ml2y_norm = (ml2[3] - ml2[1]) / length_ml2

        length_sl2 = math.sqrt(((sl2[2] - sl2[0]) ** 2) + ((sl2[3] - sl2[1]) ** 2))
        sl2x_norm = (sl2[2] - sl2[0]) / length_sl2
        sl2y_norm = (sl2[3] - sl2[1]) / length_sl2

        cos_1 = ml2x_norm * sl2x_norm + ml2y_norm * sl2y_norm
        cos_2 = -ml2y_norm * sl2x_norm + ml2x_norm * sl2y_norm
        winkel_radians = math.acos(min(math.fabs(cos_1), 1))
        w2 = np.degrees(winkel_radians)

        if cos_1 * cos_2 < 0:
            w2 = -w2
        # define rotation
        rotation = (w1 + w2) / 2

        # calculate a rotation matrix
        rotation_matrix = np.array(((np.cos(np.radians(rotation)), -np.sin(np.radians(rotation))),
                                    (np.sin(np.radians(rotation)), np.cos(np.radians(rotation)))))

        ml1x1_t = ml1[0] * rotation_matrix[0, 0] + ml1[1] * rotation_matrix[0, 1]
        ml1y1_t = ml1[0] * rotation_matrix[1, 0] + ml1[1] * rotation_matrix[1, 1]
        ml1x2_t = ml1[2] * rotation_matrix[0, 0] + ml1[3] * rotation_matrix[0, 1]
        ml1y2_t = ml1[2] * rotation_matrix[1, 0] + ml1[3] * rotation_matrix[1, 1]

        ml2x1_t = ml2[0] * rotation_matrix[0, 0] + ml2[1] * rotation_matrix[0, 1]
        ml2y1_t = ml2[0] * rotation_matrix[1, 0] + ml2[1] * rotation_matrix[1, 1]
        ml2x2_t = ml2[2] * rotation_matrix[0, 0] + ml2[3] * rotation_matrix[0, 1]
        ml2y2_t = ml2[2] * rotation_matrix[1, 0] + ml2[3] * rotation_matrix[1, 1]

        # calculate center of gravity
        centerx_ml = (ml1x1_t + ml1x2_t + ml2x1_t + ml2x2_t) / 4
        centerx_sl = (sl1[0] + sl1[2] + sl2[0] + sl2[2]) / 4

        centery_ml = (ml1y1_t + ml1y2_t + ml2y1_t + ml2y2_t) / 4
        centery_sl = (sl1[1] + sl1[3] + sl2[1] + sl2[3]) / 4

        # define translation
        translation_x = centerx_sl - centerx_ml
        translation_y = centery_sl - centery_ml

        # write result
        transformations[x, 0] = rotation
        transformations[x, 1] = translation_x
        transformations[x, 2] = translation_y


if __name__ == "__main__":

    # prepare
    center = np.array([1280/2, 720/2])  # np.array([256, 256])
    set = 2
    seed = 2000
    rng = np.random.default_rng(seed)

    # preprocess
    scene_lines, model_lines, match_id_list = load_test_set(set, "../")
    model_lines = np.array(model_lines)

    scene_lines = transform_line_batch(scene_lines, 10, [50, 0], [0, 0])
    scene_lines = np.array(scene_lines)

    indices = [[2, 1, 2, 1]]
    transformations = np.zeros((len(indices), 3))


    t = define_transformation(np.array([model_lines[2], model_lines[1]]),
                          np.array([scene_lines[2], scene_lines[1]]),
                          np.array([256, 256]))
    print(t)
    print()

    get_transformation_rigid(model_lines, scene_lines, indices, np.array([256, 256]), transformations)
    print(transformations)



