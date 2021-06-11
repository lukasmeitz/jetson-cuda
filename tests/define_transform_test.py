import cv2
import numpy as np

from modules.line_matcher.define_transform import define_transformation
from modules.optimized.optimized_math import transform_line_batch


def test_normal_dt():

    # p1x, p1y, p2x, p2y, id
    ml1 = np.array([1.0, 1.0, 10.0, 10.0, 1.0])
    ml2 = np.array([2.0, 2.0, 15.0, 15.0, 2.0])
    model_lines = np.array([ml1, ml2])

    t = np.array([10.0, 5.0])
    center_point = np.array([256, 256])
    w = 0.0

    scene_lines = transform_line_batch(model_lines, w, t, center_point)

    t = define_transformation([model_lines[0], model_lines[1], scene_lines[0], scene_lines[1]], center_point)

    print("w, t[0], t[1], scale_factor, scale_center")
    print(t)



def test_cv_at():

    # p1x, p1y, p2x, p2y, id
    ml1 = np.array([1.0, 1.0, 10.0, 10.0, 1.0])
    ml2 = np.array([2.0, 2.0, 15.0, 15.0, 2.0])
    model_lines = np.array([ml1, ml2])

    t = np.array([5.0, 5.0])
    center_point = np.array([256, 256])
    w = 0.0

    scene_lines = transform_line_batch(model_lines, w, t, center_point)

    pts1 = np.float32([ml1[0:2], ml1[2:4], ml2[0:2], ml2[2:4]])
    pts2 = np.float32([scene_lines[0, 0:2],
                       scene_lines[0, 2:4],
                       scene_lines[1, 0:2],
                       scene_lines[1, 2:4]])
    print(pts2)
    print(pts1)

    M = cv2.getPerspectiveTransform(pts1, pts2)
    print(M)


def test_linsys_dt():

    # p1x, p1y, p2x, p2y, id
    ml1 = np.array([1.0, 1.0, 10.0, 10.0, 1.0])
    ml2 = np.array([2.0, 2.0, 15.0, 15.0, 2.0])
    model_lines = np.array([ml1, ml2])

    t = np.array([10.0, 5.0])
    center_point = np.array([256, 256])
    w = 0.0

    scene_lines = transform_line_batch(model_lines, w, t, center_point)

    t = define_transformation([model_lines[0], model_lines[1], scene_lines[0], scene_lines[1]], center_point)

    print("w, t[0], t[1], scale_factor, scale_center")
    print(t)




if __name__ == "__main__":

    #test_normal_dt()
    test_cv_at()