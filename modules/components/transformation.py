
import numpy as np
import cv2

from modules.optimized.optimized_math import transform_line_batch




def define_perspective_transformation(ml1, ml2, sl1, sl2):

    source_points = []
    dest_points = []

    source_points.append(ml1[0:2])
    source_points.append(ml1[2:4])
    source_points.append(ml2[0:2])
    source_points.append(ml2[2:4])

    dest_points.append(sl1[0:2])
    dest_points.append(sl1[2:4])
    dest_points.append(sl2[0:2])
    dest_points.append(sl2[2:4])

    source_points = np.array(source_points).astype(np.float32)
    dest_points = np.array(dest_points).astype(np.float32)

    #camera_matrix = np.array([[1.20438531e+03, 0.00000000e+00, 2.56000000e+02],
    #                 [0.00000000e+00, 1.20438531e+03, 2.56000000e+02],
    #                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]).astype(np.float32)
    #distortion_coefficients = np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float32)

    return cv2.getPerspectiveTransform(source_points, dest_points)


def define_transformation_opencv(model_lines, scene_lines):

    source_points = []
    dest_points = []

    for l in model_lines:
        source_points.append(l[0:2])
        source_points.append(l[2:4])
    for s in scene_lines:
        dest_points.append(s[0:2])
        dest_points.append(s[2:4])

    source_points = np.array(source_points[:3]).astype(np.float32)
    dest_points = np.array(dest_points[:3]).astype(np.float32)

    transform_matrix = cv2.getAffineTransform(source_points, dest_points)

    return transform_matrix


def define_transformation_pair_opencv(ml1, ml2, sl1, sl2):

    p = np.array([ml1[0:2], ml1[2:4], ml2[2:4]]).astype(np.float32)
    p_prime = np.array([sl1[0:2], sl1[2:4], sl2[2:4]]).astype(np.float32)

    return cv2.getAffineTransform(p, p_prime)


def define_transformation_pair_midpoint_opencv(ml1, ml2, sl1, sl2):
    # create ml triangle
    p3m = calc_intersection(ml1, ml2)

    if calc_distance(p3m, ml1[0:2]) > calc_distance(p3m, ml1[2:4]):
        p1m = (ml1[0], ml1[1])
    else:
        p1m = (ml1[2], ml1[3])

    if calc_distance(p3m, ml2[0:2]) > calc_distance(p3m, ml2[2:4]):
        p2m = (ml2[0], ml2[1])
    else:
        p2m = (ml2[2], ml2[3])

    p = np.array((p1m, p2m, (p3m[0], p3m[1]))).astype(np.float32)

    # create sl triangle
    p3s = calc_intersection(sl1, sl2)

    if calc_distance(p3s, sl1[0:2]) > calc_distance(p3s, sl1[2:4]):
        p1s = (sl1[0], sl1[1])
    else:
        p1s = (sl1[2], sl1[3])

    if calc_distance(p3s, sl2[0:2]) > calc_distance(p3s, sl2[2:4]):
        p2s = (sl2[0], sl2[1])
    else:
        p2s = (sl2[2], sl2[3])

    p_prime = np.array((p1s, p2s, (p3s[0], p3s[1]))).astype(np.float32)

    return cv2.getAffineTransform(p, p_prime)


def define_transformation_numpy_pairs(ml1, ml2, sl1, sl2):

    # create ml triangle
    p3m = calc_intersection(ml1, ml2)

    if calc_distance(p3m, ml1[0:2]) > calc_distance(p3m, ml1[2:4]):
        p1m = (ml1[0], ml1[1], 0.0)
    else:
        p1m = (ml1[2], ml1[3], 0.0)

    if calc_distance(p3m, ml2[0:2]) > calc_distance(p3m, ml2[2:4]):
        p2m = (ml2[0], ml2[1], 0.0)
    else:
        p2m = (ml2[2], ml2[3], 0.0)

    p = np.array((p1m, p2m, (p3m[0], p3m[1], 0.0)))


    # create sl triangle
    p3s = calc_intersection(sl1, sl2)

    if calc_distance(p3s, sl1[0:2]) > calc_distance(p3s, sl1[2:4]):
        p1s = (sl1[0], sl1[1], 0.0)
    else:
        p1s = (sl1[2], sl1[3], 0.0)

    if calc_distance(p3s, sl2[0:2]) > calc_distance(p3s, sl2[2:4]):
        p2s = (sl2[0], sl2[1], 0.0)
    else:
        p2s = (sl2[2], sl2[3], 0.0)

    p_prime = np.array((p1s, p2s, (p3s[0], p3s[1], 0.0)))

    return define_transformation_numpy(p, p_prime)


def define_transformation_numpy(p, p_prime):

    '''
        Find the unique homogeneous affine transformation that
        maps a set of 3 points to another set of 3 points in 3D
        space:

            p_prime == np.dot(p, R) + t

        where `R` is an unknown rotation matrix, `t` is an unknown
        translation vector, and `p` and `p_prime` are the original
        and transformed set of points stored as row vectors:

            p       = np.array((p1,       p2,       p3))
            p_prime = np.array((p1_prime, p2_prime, p3_prime))

        The result of this function is an augmented 4-by-4
        matrix `A` that represents this affine transformation:

            np.column_stack((p_prime, (1, 1, 1))) == \
                np.dot(np.column_stack((p, (1, 1, 1))), A)

        Source: https://math.stackexchange.com/a/222170 (robjohn)
    '''

    # construct intermediate matrix
    Q = p[1:] - p[0]
    Q_prime = p_prime[1:] - p_prime[0]

    # calculate rotation matrix
    R = np.dot(
        np.linalg.inv(
            np.row_stack(
                (Q,
                 np.cross(*Q))
            )
        ),

        np.row_stack(
            (Q_prime,
             np.cross(*Q_prime))
        )
    )

    # calculate translation vector
    t = p_prime[0] - np.dot(p[0], R)

    # calculate affine transformation matrix
    return np.column_stack((np.row_stack((R, t)),
                            (0, 0, 0, 1)))



def define_transformation(model_line_pair, scene_line_pair, center_point):

    # pair = [line1, line2]

    # berechne ursprungs-einheitsvektoren aus den modelllinien und scenelinien
    ml1 = np.array([ model_line_pair[0, 2] - model_line_pair[0, 0],
            model_line_pair[0, 3] - model_line_pair[0, 1]])
    ml1 = ml1 / np.linalg.norm(ml1)

    sl1 = np.array([ scene_line_pair[0, 2] - scene_line_pair[0, 0],
            scene_line_pair[0, 3] - scene_line_pair[0, 1]])
    sl1 = sl1 / np.linalg.norm(sl1)
    w1 = calc_angle(ml1, sl1)

    ml2 = np.array([model_line_pair[1, 2] - model_line_pair[1, 0],
           model_line_pair[1, 3] - model_line_pair[1, 1]])
    ml2 = ml2 / np.linalg.norm(ml2)
    sl2 = np.array([ scene_line_pair[1, 2] - scene_line_pair[1, 0],
            scene_line_pair[1, 3] - scene_line_pair[1, 1]])
    sl2 = sl2 / np.linalg.norm(sl2)
    w2 = calc_angle(ml2, sl2)

    w =  (w1 + w2) / 2 # beware the negative sign! i removed it


    # correction of modellines
    model_line_pair_rotated = transform_line_batch(model_line_pair, w, np.array([0,0]), center_point)
    m1 = model_line_pair_rotated[0]
    m2 = model_line_pair_rotated[1]


    # regular scenelines 1 and 2
    s1 = scene_line_pair[0]
    s2 = scene_line_pair[1]

    # compensate translation
    w5 = abs(calc_angle(ml1, ml2))

    if w5 > 20:
        # if there is enough angle just find a mean of the translation
        a = calc_intersection(m1, m2)
        b = calc_intersection(s1, s2)
        t = [b[0] - a[0], b[1] - a[1]]
        scale_center = b
        scale_factor = 1  # there is no reasonable factor to calculate at this point

    else:
        tx = np.mean(np.array([m1[0] - s1[0], m1[2] - s1[2], m2[0] - s2[0], m2[2] - s2[2]]))
        ty = np.mean(np.array([m1[1] - s1[1], m1[3] - s1[3], m2[1] - s2[1], m2[3] - s2[3]]))
        t = [-tx, -ty]

        scale_center = np.array([(m1[0] + m1[2]) / 2, (m1[1] + m1[3]) / 2])
        scale_factor = 1  # there is no reasonable factor to calculate at this point

    return [get_rotation_matrix_2d(w), t[0], t[1], scale_factor, scale_center]


def get_rotation_matrix_2d(rotation_angle):

    # calculate a rotation matrix
    r = np.array(((np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))),
                  (np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle)))))

    return r


def calc_angle(a, b):

    cos_winkel_1 = a[0] * b[0] + a[1] * b[1]
    cos_winkel_2 = -a[1] * b[0] + a[0] * b[1]
    winkel_radians = np.arccos(min(abs(cos_winkel_1), 1))
    winkel_grad = np.degrees(winkel_radians)

    if cos_winkel_1 * cos_winkel_2 < 0:
        winkel_grad = -winkel_grad

    return winkel_grad


def calc_distance(a, b):

    x_dist = b[0] - a[0]
    y_dist = b[1] - a[1]
    dist = np.sqrt(x_dist ** 2 + y_dist ** 2)

    return dist


def calc_intersection(line1, line2):

    l1x = line1[0]
    l1y = line1[1]

    l1_xdiff = line1[2] - l1x
    l1_ydiff = line1[3] - l1y

    l2x = line2[0]
    l2y = line2[1]

    l2_xdiff = line2[2] - l2x
    l2_ydiff = line2[3] - l2y

    yy = (l2y - l1y - ((l2x / l1_xdiff) * l1_ydiff) + ((l1x / l1_xdiff) * l1_ydiff))
    #print("yy: " + str(yy))
    xx = (l2_xdiff / l1_xdiff * l1_ydiff - l2_ydiff)
    #print("xx: " + str(xx))

    if xx == 0:
        y = 0
    else:
        y = yy / xx

    return [l2x, l2y] + [y * l2_xdiff, y * l2_ydiff]