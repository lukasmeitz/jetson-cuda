
import math
import scipy.io
import numpy as np

from numba import cuda



def load_test_set(n, path, demo=False):

    # load the file
    test_set_number = "{:03d}".format(n)
    if not demo:
        test_set_data = scipy.io.loadmat(path + 'data/TestSets/TestSet' + test_set_number + '/TestSet' + test_set_number + '.mat')
    else:
        test_set_data = scipy.io.loadmat(
        path + 'data/TestSets_Demonstrator/TestSet' + test_set_number + '/TestSet' + test_set_number + '.mat')

    #print(test_set_data)

    # save the sceneline array
    # fields of one vector: p1, p2, vec, mid, len, ang
    scene_lines = test_set_data['Results_t']['Scenelines'][0][:][0][0]

    # save the modelline array
    # fields of one vector: p1, p2, vec, mid, len, ang
    model_lines = test_set_data['Results_t']['Modellines_err'][0][:][0][0]

    # get rid of unnecessary data
    scene_lines = [[p1[0][0], p1[1][0], p2[0][0], p2[1][0], ang[0][0], len[0][0], mid] for p1, p2, vec, mid, len, ang in scene_lines]
    model_lines = [[p1[0][0], p1[1][0], p2[0][0], p2[1][0], ang[0][0], len[0][0], mid] for p1, p2, vec, _, len, ang, _, mid, _, _, _ in model_lines]

    # give ids to lines
    scene_lines = [[line[0], line[1], line[2], line[3], line[4], line[5], int(num)] for num, line in enumerate(scene_lines)]
    model_lines = [[line[0], line[1], line[2], line[3], line[4], line[5], int(num)] for num, line in enumerate(model_lines)]

    # create numpy array
    scene_lines = np.array(scene_lines)
    model_lines = np.array(model_lines)

    # save the test set meta data
    match_ids = test_set_data['matchingOutput']['Matches'][0][:][0][0]

    return scene_lines, model_lines, match_ids

@cuda.jit
def get_transformation_cuda(model_lines, scene_lines, indices, transformations):

    x = cuda.grid(1)

    if x < indices.shape[0]:
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
        w1 = math.degrees(winkel_radians)

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
        w2 = math.degrees(winkel_radians)

        if cos_1 * cos_2 < 0:
            w2 = -w2
        # define rotation
        rotation = (w1 + w2) / 2

        # calculate a rotation matrix
        rot_00 = math.cos(math.radians(rotation))
        rot_01 = -math.sin(math.radians(rotation))
        rot_10 = math.sin(math.radians(rotation))
        rot_11 = math.cos(math.radians(rotation))


        ml1x1_t = ml1[0] * rot_00 + ml1[1] * rot_01
        ml1y1_t = ml1[0] * rot_10 + ml1[1] * rot_11
        ml1x2_t = ml1[2] * rot_00 + ml1[3] * rot_01
        ml1y2_t = ml1[2] * rot_10 + ml1[3] * rot_11

        ml2x1_t = ml2[0] * rot_00 + ml2[1] * rot_01
        ml2y1_t = ml2[0] * rot_10 + ml2[1] * rot_11
        ml2x2_t = ml2[2] * rot_00 + ml2[3] * rot_01
        ml2y2_t = ml2[2] * rot_10 + ml2[3] * rot_11

        # calculate center of gravity
        centerx_ml = (ml1x1_t + ml1x2_t + ml2x1_t + ml2x2_t) / 4
        centerx_sl = (sl1[0] + sl1[2] + sl2[0] + sl2[2]) / 4

        centery_ml = (ml1y1_t + ml1y2_t + ml2y1_t + ml2y2_t) / 4
        centery_sl = (sl1[1] + sl1[3] + sl2[1] + sl2[3]) / 4

        # define translation
        translation_x = centerx_sl - centerx_ml
        translation_y = centery_sl - centery_ml

        # write result
        transformations[x, 0] = rotation < 30 and math.fabs(translation_x) + math.fabs(translation_y) < 150
        transformations[x, 1] = rotation
        transformations[x, 2] = translation_x
        transformations[x, 3] = translation_y



@cuda.jit
def iterate_2D_array(an_array, result):

    result = 999.0
    i = cuda.grid(1)

    if i < an_array.shape[0]:
        if result < an_array[i]:
            result = an_array[i]


@cuda.jit
def evaluate_line_batch(lines, reference_lines, scores, threshold):

    m, s = cuda.grid(2)
    if m < lines.shape[0] and s < reference_lines.shape[0]:

        # line1 point 1 to line2 distance
        dist_11 = math.sqrt(
            (reference_lines[s][0] - lines[m][0]) ** 2
            + (reference_lines[s][1] - lines[m][1]) ** 2)

        dist_12 = math.sqrt(
            (reference_lines[s][2] - lines[m][0]) ** 2
            + (reference_lines[s][3] - lines[m][1]) ** 2)

        if dist_11 < dist_12:
            dist_2 = math.sqrt(
                (reference_lines[s][2] - lines[m][2]) ** 2
                + (reference_lines[s][3] - lines[m][3]) ** 2)

            scores[((lines.shape[0]) * s) + m] = (dist_11 + dist_2) / 2
        else:
            dist_2 = math.sqrt(
                (reference_lines[s][0] - lines[m][2]) ** 2
                + (reference_lines[s][1] - lines[m][3]) ** 2)
            scores[((lines.shape[0]) * s) + m] = (dist_12 + dist_2) / 2

        scores[((lines.shape[0]) * s) + m] = min(scores[((lines.shape[0]) * s) + m], threshold)


@cuda.jit
def transform_line_batch(lines, rotation, transformation_distance, center_point):

    x = cuda.grid(1)
    if x < lines.shape[0]:

        # translate p1 to origin
        lines[x, 0] -= center_point[0]
        lines[x, 1] -= center_point[1]
        # rotate
        lines[x, 0] = lines[x, 0] * rotation[0, 0] + lines[x, 1] * rotation[0, 1]
        lines[x, 1] = lines[x, 0] * rotation[1, 0] + lines[x, 1] * rotation[1, 1]
        # translate forth
        lines[x, 0] += center_point[0] + transformation_distance[0]
        lines[x, 1] += center_point[1] + transformation_distance[1]

        # translate p2 to origin
        lines[x, 2] -= center_point[0]
        lines[x, 3] -= center_point[1]
        # rotate
        lines[x, 2] = lines[x, 2] * rotation[0, 0] + lines[x, 3] * rotation[0, 1]
        lines[x, 3] = lines[x, 2] * rotation[1, 0] + lines[x, 3] * rotation[1, 1]
        # translate forth
        lines[x, 2] += center_point[0] + transformation_distance[0]
        lines[x, 3] += center_point[1] + transformation_distance[1]


@cuda.reduce
def count_inlier_error(a, b):
    return a + b


def ransac_cuda_final(model_lines,
                      scene_lines,
                      random_generator,
                      center,
                      threshold=40,
                      iterations=500):

    # Parameters
    max_inliers = 0
    best_error = 99999999
    best_transformation = []
    best_matches = []

    indices = random_generator.random((iterations, 4))
    indices *= [len(model_lines) - 1,
                len(model_lines) - 1,
                len(scene_lines) - 1,
                len(scene_lines) - 1]
    indices = np.round(indices).astype(int)

    transformations = np.zeros((len(indices), 4))

    # build hypotheses
    threadsperblock = (16, 16)
    blockspergrid_x = np.math.ceil(len(indices) / threadsperblock[0])
    blockspergrid_y = np.math.ceil(len(scene_lines) / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    get_transformation_cuda[blockspergrid, threadsperblock](model_lines, scene_lines, indices, transformations)

    # loop through
    for i in range(iterations):

        if not transformations[i][0]:
            continue

        # batch transformation
        model_lines_transformed = np.copy(model_lines)
        threadsperblock = (16, 16)
        blockspergrid_x = np.math.ceil(model_lines.shape[0] / threadsperblock[0])
        blockspergrid_y = np.math.ceil(model_lines.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        rotation_matrix = np.array(((np.cos(np.radians(transformations[i][1])),
                                    -np.sin(np.radians(transformations[i][1]))),
                                    (np.sin(np.radians(transformations[i][1])),
                                     np.cos(np.radians(transformations[i][1])))))
        transform_line_batch[blockspergrid, threadsperblock](model_lines_transformed,
                                                             rotation_matrix,
                                                             np.array([transformations[i][2],
                                                                      transformations[i][2]]),
                                                             center)

        # evaluation
        results = np.zeros((len(model_lines_transformed) * len(scene_lines)))
        blockspergrid_x = np.math.ceil(len(model_lines_transformed) / threadsperblock[0])
        blockspergrid_y = np.math.ceil(len(scene_lines) / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        evaluate_line_batch[blockspergrid, threadsperblock](model_lines_transformed, scene_lines, results, threshold)

        error_cuda = count_inlier_error(results)

        if error_cuda < best_error:
            best_error = error_cuda
            best_transformation = transformations[i]

            best_matches.clear()
            max_inliers = 0

            c = 0
            for n, r in enumerate(results):

                # id generation: scores[((lines.shape[0]) * s) + m]

                # this means inlier
                if r < threshold:
                    m = c % len(model_lines)
                    s = np.floor(c / len(model_lines))
                    best_matches.append((int(model_lines[int(m)][6]),
                                         int(scene_lines[int(s)][6])))
                    max_inliers += 1

                c += 1

    return best_matches, best_transformation


if __name__ == "__main__":

    # prepare
    center = np.array([256, 256])
    set = 37
    seed = 2001
    rng = np.random.default_rng(seed)

    # preprocess
    scene_lines, model_lines, match_id_list = load_test_set(set, "../../")


    #scene_lines = preprocess_length(scene_lines, max_lines=300)
    #model_lines = preprocess_length(model_lines, max_lines=300)
    model_lines = np.array(model_lines)
    scene_lines = np.array(scene_lines)


    # sample
    matches, transform = ransac_cuda_final(model_lines,
                                           scene_lines,
                                           rng,
                                           center,
                                           threshold=15,
                                           iterations=10000)
    print(transform)
    print(matches)
