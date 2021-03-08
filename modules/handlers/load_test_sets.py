import numpy as np
import scipy.io

from modules.visuals.imaging import draw_lines, plot_image


def load_test_set(n, path):

    # load the file
    test_set_number = "{:03d}".format(n)
    test_set_data = scipy.io.loadmat(path + 'data/TestSets/TestSet' + test_set_number + '/TestSet' + test_set_number + '.mat')

    # save the sceneline array
    # fields of one vector: p1, p2, vec, mid, len, ang
    scene_lines = test_set_data['Results_t']['Scenelines'][0][:][0][0]

    # save the modelline array
    # fields of one vector: p1, p2, vec, mid, len, ang
    model_lines = test_set_data['Results_t']['Modellines_err'][0][:][0][0]

    # save the test set meta data
    match_ids = test_set_data['matchingOutput']['Matches']

    return scene_lines, model_lines, match_ids


if __name__ == "__main__":

    a1, a2, a3 = load_test_set(24, "../../")
    print(a3)
