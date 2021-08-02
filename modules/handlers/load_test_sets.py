import numpy as np
import scipy.io
import itertools

from modules.visuals.imaging import draw_lines, plot_image


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
    scene_lines = [[p1[0][0], p1[1][0], p2[0][0], p2[1][0], mid, ang[0][0], len[0][0]] for p1, p2, vec, mid, len, ang in scene_lines]
    model_lines = [[p1[0][0], p1[1][0], p2[0][0], p2[1][0], mid, ang[0][0], len[0][0]] for p1, p2, vec, _, len, ang, _, mid, _, _, _ in model_lines]

    # give ids to lines
    scene_lines = [[line[0], line[1], line[2], line[3], line[4], line[5], line[6], num] for num, line in enumerate(scene_lines)]
    model_lines = [[line[0], line[1], line[2], line[3], line[4], line[5], line[6], num] for num, line in enumerate(model_lines)]

    # create numpy array
    scene_lines = np.array(scene_lines)
    model_lines = np.array(model_lines)

    # save the test set meta data
    match_ids = test_set_data['matchingOutput']['Matches'][0][:][0][0]

    return scene_lines, model_lines, match_ids


def create_line_permutations(line_array):

    permutations = [[l1, l2] for l1, l2 in list(itertools.combinations(line_array, r=2))]

    return np.array(permutations)


if __name__ == "__main__":

    a1, a2, a3 = load_test_set(24, "../../")
    print(a3)
