import numpy as np
import scipy.io


# out:  model_linee_pairs, scene_line_pairs
def load_lines_data():

    # vectors for return
    model_line_pairs = []
    scene_line_pairs = []

    # read matlab file and convert to numpy array
    line_pair_data = scipy.io.loadmat('../../data/line_matcher_input/listeModScene.mat')
    line_pair_data = line_pair_data['listeModScene']

    # read modellines
    model_lines = scipy.io.loadmat('../../data/line_matcher_input/modellines.mat')
    model_lines = np.array(model_lines["modellines"][0][:])

    # and the scenelines
    scene_lines = scipy.io.loadmat('../../data/line_matcher_input/scenelines.mat')
    scene_lines = np.array(scene_lines["scenelines"][0][:])


    for num in range(len(line_pair_data[0])):

        # resolve indices for real data
        # to achieve this we need to take the following, cruel steps:
        # 1) de-reference the numpy array by taking line_pair_data[0]
        # 2) take the current element by indexing with our loop-counter num
        # 3) subtract 1 off the line index, because matlab does not know how to computer science
        # 4) take the first and second elements in the resulting list, which is p1 and p2
        # 5) again, dereference the inner numpy array to get a float value instead of a 1x1 vector
        # 6) easy breezy, here is your point data consisting of two floats for x and y position!

        # model lines 1 and 2
        ml1_p1 = model_lines[line_pair_data[0][num]-1][0].astype(int)
        ml1_p1 = [ml1_p1[0][0], ml1_p1[1][0]]
        ml1_p2 = model_lines[line_pair_data[0][num]-1][1].astype(int)
        ml1_p2 = [ml1_p2[0][0], ml1_p2[1][0]]
        ml1 = [ml1_p1[0], ml1_p1[1], ml1_p2[0], ml1_p2[1]]

        ml2_p1 = model_lines[line_pair_data[2][num]-1][0].astype(int)
        ml2_p1 = [ml2_p1[0][0], ml2_p1[1][0]]
        ml2_p2 = model_lines[line_pair_data[2][num]-1][1].astype(int)
        ml2_p2 = [ml2_p2[0][0], ml2_p2[1][0]]
        ml2 = [ml2_p1[0], ml2_p1[1], ml2_p2[0], ml2_p2[1]]

        model_line_pairs += [[ml1, ml2]]

        # scene lines 1 and 2
        sl1_p1 = scene_lines[line_pair_data[1][num]-1][0].astype(int)
        sl1_p1 = [sl1_p1[0][0], sl1_p1[1][0]]
        sl1_p2 = scene_lines[line_pair_data[1][num]-1][1].astype(int)
        sl1_p2 = [sl1_p2[0][0], sl1_p2[1][0]]
        sl1 = [sl1_p1[0], sl1_p1[1], sl1_p2[0], sl1_p2[1]]

        sl2_p1 = scene_lines[line_pair_data[3][num]-1][0].astype(int)
        sl2_p1 = [sl2_p1[0][0], sl2_p1[1][0]]
        sl2_p2 = scene_lines[line_pair_data[3][num]-1][1].astype(int)
        sl2_p2 = [sl2_p2[0][0], sl2_p2[1][0]]
        sl2 = [sl2_p1[0], sl2_p1[1], sl2_p2[0], sl2_p2[1]]

        scene_line_pairs += [[sl1, sl2]]

    return model_line_pairs, scene_line_pairs
