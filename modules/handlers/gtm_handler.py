import numpy as np

def read_gtm_file(path):

    # open the file to read content
    gtm_file = open(path, 'r')
    lines = gtm_file.readlines()

    # read left keypoints
    # singleKeyPoint.pt.x, singleKeyPoint.pt.y, singleKeyPoint.response,
    #   singleKeyPoint.angle, singleKeyPoint.size, singleKeyPoint.octave,
    #   singleKeyPoint.class_id
    key_points_left = lines[9].split(' ')[1:-1]
    key_points_left = np.array(key_points_left).reshape(len(key_points_left)//7, 7)
    print(key_points_left)

    # read right keypoints
    key_points_right = lines[10].split(' ')[1:-1]
    key_points_right = np.array(key_points_right).reshape(len(key_points_right) // 7, 7)
    print(key_points_right)


    # read matches
    # query_id, train_id, distance
    matches = lines[8].split(' ')[1:-1]
    matches = np.array(matches).reshape(len(matches)//3, 3)
    #print(matches)

    # read inliers
    # match_id, boolean
    inliers = lines[7].split(' ')[1:-1]
    inliers = np.array(inliers) #.reshape(len(matches)//3, 3)
    #print(inliers)

    return key_points_left, key_points_right, matches, inliers
