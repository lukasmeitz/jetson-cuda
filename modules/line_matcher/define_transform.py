import numpy as np
from numba import jit

# in: line1 = [p1x, p1y, p2x, p2y, id]
# in: line2 = [p1x, p1y, p2x, p2y, id]
def calc_min_pair_distance(ml1, ml2, sl1, sl2):

    error = 999

    # distance of all points to one another
    dist_11 = calc_min_line_distance(ml1, sl1)
    dist_12 = calc_min_line_distance(ml1, sl2)

    if dist_11 < dist_12:
        dist_2 = calc_min_line_distance(ml2, sl2)
        error = np.sqrt((dist_11 ** 2) + (dist_2 ** 2))
    else:
        dist_2 = calc_min_line_distance(ml2, sl1)
        error = np.sqrt((dist_11 ** 2) + (dist_2 ** 2))

    return error


def calc_min_line_distance(line1, line2):

    error = 999

    # line1 point 1 to line2 distance
    dist_11 = calc_distance(line1[0:2], line2[0:2])
    dist_12 = calc_distance(line1[0:2], line2[2:4])

    # if l1p1 to l2p1 is closest
    if dist_11 < dist_12:
        dist_2 = calc_distance(line1[2:4], line2[2:4])
        error = np.sqrt((dist_11 ** 2) + (dist_2 ** 2))
    else:
        dist_2 = calc_distance(line1[2:4], line2[0:2])
        error = np.sqrt((dist_12 ** 2) + (dist_2 ** 2))

    return error



# in: line1 = [p1x, p1y, p2x, p2y, id]
# in: line2 = [p1x, p1y, p2x, p2y, id]
def calc_line_distance(ml1, ml2, sl1, sl2):

    dist_m1s1 = np.sqrt(
        calc_distance(ml1[0:2], sl1[0:2]) ** 2
        + calc_distance(ml1[2:4], sl1[2:4]) ** 2
    )

    dist_m1s2 = np.sqrt(
        calc_distance(ml1[0:2], sl2[0:2]) ** 2
        + calc_distance(ml1[2:4], sl2[2:4]) ** 2
    )

    dist_m2s1 = np.sqrt(
        calc_distance(ml2[0:2], sl1[0:2]) ** 2
        + calc_distance(ml2[2:4], sl1[2:4]) ** 2
    )

    dist_m2s2 = np.sqrt(
        calc_distance(ml2[0:2], sl2[0:2]) ** 2
        + calc_distance(ml2[2:4], sl2[2:4]) ** 2
    )

    return np.mean([dist_m1s1, dist_m1s2, dist_m2s1, dist_m2s2])


# distance between two points
# in:   a = [x, y]
# in:   b = [x, y]
def calc_distance(a, b):

    x_dist = b[0] - a[0]
    y_dist = b[1] - a[1]
    dist = np.sqrt(x_dist ** 2 + y_dist ** 2)

    return dist

# angle between two vectors
# in:   a = [x, y]
# in:   b = [x, y]
def calc_angle(a, b):

    cos_winkel_1 = a[0] * b[0] + a[1] * b[1]
    cos_winkel_2 = -a[1] * b[0] + a[0] * b[1]
    winkel_radians = np.arccos(min(abs(cos_winkel_1), 1))
    winkel_grad = np.degrees(winkel_radians)

    if cos_winkel_1 * cos_winkel_2 < 0:
        winkel_grad = -winkel_grad

    return winkel_grad


# in:   linie1 = [p1x p1y p2x p2y]
#       linie2 = [p1x p1y p2x p2y]
def calc_intersection(line1, line2):

    l1x = line1[0]
    l1y = line1[1]

    l1_xdiff = line1[2] - l1x
    l1_ydiff = line1[3] - l1y

    l2x = line2[0]
    l2y = line2[1]

    l2_xdiff = line2[2] - l2x
    l2_ydiff = line2[3] - l2y

    y = (l2y - l1y - l2x / l1_xdiff * l1_ydiff + l1x / l1_xdiff * l1_ydiff) / (l2_xdiff / l1_xdiff * l1_ydiff - l2_ydiff)

    return [l2x, l2y] + [y * l2_xdiff, y * l2_ydiff]


def get_rotation_matrix_2d(rotation_angle):

    # calculate a rotation matrix
    r = np.array(((np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))),
                  (np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle)))))

    return r


def transform_line_batch(lines, rotation_angle, transformation_distance, center_point):

    # return variables
    lines_transformed = []

    # rotation matrix
    r = get_rotation_matrix_2d(rotation_angle)

    for i in range(len(lines)):

        # correction of modelline 1
        m1_p1 = lines[i][0:2]
        m1_p2 = lines[i][2:4]
        m1_p1_corrected = r.dot([m1_p1[0] - center_point[0], m1_p1[1] - center_point[1]]) + center_point
        m1_p2_corrected = r.dot([m1_p2[0] - center_point[0], m1_p2[1] - center_point[1]]) + center_point
        m1 = [m1_p1_corrected[0] + transformation_distance[0],
              m1_p1_corrected[1] + transformation_distance[1],
              m1_p2_corrected[0] + transformation_distance[0],
              m1_p2_corrected[1] + transformation_distance[1],
              lines[i][4]]

        lines_transformed += [m1]

    return lines_transformed


# in:   model_pairs = [modellines1, modellines2]
#       rotation_angle = angle in degrees
#       transformation distance = [distance_x, distance_y]
#       center_point = [x y]
def transform_modelline_batch(model_pairs, rotation_angle, transformation_distance, center_point):

    # return variables
    model_pairs_transformed = []

    # rotation matrix
    r = get_rotation_matrix_2d(rotation_angle)

    for i in range(len(model_pairs)):

        # correction of modelline 1
        m1_p1 = model_pairs[i][0][0:2]
        m1_p2 = model_pairs[i][0][2:4]
        m1_p1_corrected = r.dot([m1_p1[0] - center_point[0], m1_p1[1] - center_point[1]]) + center_point
        m1_p2_corrected = r.dot([m1_p2[0] - center_point[0], m1_p2[1] - center_point[1]]) + center_point
        m1 = [m1_p1_corrected[0] + transformation_distance[0],
              m1_p1_corrected[1] + transformation_distance[1],
              m1_p2_corrected[0] + transformation_distance[0],
              m1_p2_corrected[1] + transformation_distance[1],
              model_pairs[i][0][4]]

        # correction of modelline 2
        m2_p1 = model_pairs[i][1][0:2]
        m2_p2 = model_pairs[i][1][2:4]
        m2_p1_corrected = r.dot([m2_p1[0] - center_point[0], m2_p1[1] - center_point[1]]) + center_point
        m2_p2_corrected = r.dot([m2_p2[0] - center_point[0], m2_p2[1] - center_point[1]]) + center_point
        m2 = [m2_p1_corrected[0] + transformation_distance[0],
              m2_p1_corrected[1] + transformation_distance[1],
              m2_p2_corrected[0] + transformation_distance[0],
              m2_p2_corrected[1] + transformation_distance[1],
              model_pairs[i][1][4]]

        model_pairs_transformed += [[m1, m2]]

    return model_pairs_transformed


# in: modelline1 = [p1x p1y p2x p2y]
# in: X = [modelline1, modelline2, sceneline1, sceneline2]
def define_transformation(X, CoP):

    P = []
    scale_factor = 1
    scale_center = [0, 0]

    # get rotation angle
    # berechne ursprungs-einheitsvektoren aus den modelllinien und scenelinien
    ml1 = [ X[0][2] - X[0][0], X[0][3] - X[0][1] ]
    ml1 = ml1 / np.linalg.norm(ml1)
    sl1 = [ X[2][2] - X[2][0], X[2][3] - X[2][1] ]
    sl1 = sl1 / np.linalg.norm(sl1)
    w1 = calc_angle(ml1, sl1)

    ml2 = [ X[1][2] - X[1][0], X[1][3] - X[1][1] ]
    ml2 = ml2 / np.linalg.norm(ml2)
    sl2 = [ X[3][2] - X[3][0], X[3][3] - X[3][1] ]
    sl2 = sl2 / np.linalg.norm(sl2)
    w2 = calc_angle(ml2, sl2)

    w = (w1 + w2) / 2 # beware the negative sign! i removed it

    # compensate rotation
    r = np.array(((np.cos(np.radians(w)), -np.sin(np.radians(w))),
                  (np.sin(np.radians(w)), np.cos(np.radians(w)))))

    # correction of modelline 1
    m1_p1 = X[0][0:2]
    m1_p2 = X[0][2:4]
    m1_p1_corrected = r.dot([m1_p1[0] - CoP[0], m1_p1[1] - CoP[1]]) + CoP
    m1_p2_corrected = r.dot([m1_p2[0] - CoP[0], m1_p2[1] - CoP[1]]) + CoP
    m1 = [m1_p1_corrected[0], m1_p1_corrected[1], m1_p2_corrected[0], m1_p2_corrected[1]]

    # correction of modelline 2
    m2_p1 = X[1][0:2]
    m2_p2 = X[1][2:4]
    m2_p1_corrected = r.dot([m2_p1[0] - CoP[0], m2_p1[1] - CoP[1]]) + CoP
    m2_p2_corrected = r.dot([m2_p2[0] - CoP[0], m2_p2[1] - CoP[1]]) + CoP
    m2 = [m2_p1_corrected[0], m2_p1_corrected[1], m2_p2_corrected[0], m2_p2_corrected[1]]

    # regular scenelines 1 and 2
    s1 = X[2]
    s2 = X[3]

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
        tx = np.mean([m1[0] - s1[0], m1[2] - s1[2], m2[0] - s2[0], m2[2] - s2[2]])
        ty = np.mean([m1[1] - s1[1], m1[3] - s1[3], m2[1] - s2[1], m2[3] - s2[3]])
        t = [-tx, -ty]

        scale_center = [(m1[0] + m1[2]) / 2, (m1[1] + m1[3]) / 2]
        scale_factor = 1  # there is no reasonable factor to calculate at this point

    return [w, t[0], t[1], scale_factor, scale_center]  # , m1, m2]

    '''
    else:

        # get rotation angle
        ml1 = [X(3, 1) X(4, 1)] - [X(1, 1) X(2, 1)]
        ml1 = ml1. / norm(ml1);
        sl1 = [X(7, 1) X(8, 1)] - [X(5, 1) X(6, 1)]
        sl1 = sl1. / norm(sl1)
        w1 = calc_angle(ml1, sl1)

        ml2 = [X(3, 2) X(4, 2)] - [X(1, 2) X(2, 2)]
        ml2 = ml2. / norm(ml2)
        sl2 = [X(7, 2) X(8, 2)] - [X(5, 2) X(6, 2)]
        sl2 = sl2. / norm(sl2)
        w2 = calc_angle(ml2, sl2)

        ml3 = [X(3, 3) X(4, 3)] - [X(1, 3) X(2, 3)]
        ml3 = ml3. / norm(ml3)
        sl3 = [X(7, 3) X(8, 3)] - [X(5, 3) X(6, 3)]
        sl3 = sl3. / norm(sl3)
        w3 = calc_angle(ml3, sl3)

        w = -(w1 + w2 + w3) / 3

        # compensate rotation
        R = RotationMatrix2D(w)

        m1.p1 = R * (X(1:2, 1) - CoP)+CoP;
        m1.p2 = R * (X(3:4, 1) - CoP)+CoP;
        s1.p1 = X(5:6, 1);
        s1.p2 = X(7:8, 1);

        m2.p1 = R * (X(1:2, 2) - CoP)+CoP;
        m2.p2 = R * (X(3:4, 2) - CoP)+CoP;
        s2.p1 = X(5:6, 2);
        s2.p2 = X(7:8, 2);

        m3.p1 = R * (X(1:2, 3) - CoP)+CoP;
        m3.p2 = R * (X(3:4, 3) - CoP)+CoP;
        s3.p1 = X(5:6, 3);
        s3.p2 = X(7:8, 3);

        # compensate translation
        w5 = abs(calc_angle(ml1, ml2));
        if (w5 > 20):
            # if there is enough angle just find a mean of the translation
            a5 = calcIntersection(m1, m2);
            b5 = calcIntersection(s1, s2);
            t1 = a5 - b5;
            u1 = 1;
        else:
            a5 = (m1.p1' + m1.p2') / 2;
            b5 = (s1.p1 + s1.p2) / 2;
            t1 = 0;
            u1 = 0;

        w6 = abs(calc_angle(ml1, ml3));
        if (w6 > 20):
            # if there is enough angle just find a mean of the translation
            a6 = calcIntersection(m1, m3);
            b6 = calcIntersection(s1, s3);
            t2 = a6 - b6;
            u2 = 1;
        else:
            a6 = (m1.p1' + m3.p2') / 2;
            b6 = (s1.p1' + s3.p2') / 2;
            t2 = 0;
            u2 = 0;

        w7 = abs(calc_angle(ml2, ml3));
        if (w7 > 20):
            # if there is enough angle just find a mean of the translation
            a7 = calcIntersection(m2, m3);
            b7 = calcIntersection(s2, s3);
            t3 = a7 - b7;
            u3 = 1;
        else:
            a7 = (m2.p1' + m3.p2') / 2;
            b7 = (s2.p1' + s3.p2') / 2;
            t3 = 0;
            u3 = 0;

        if (u1 == 1) && (u2 == 1) && (u3 == 1):
            t = -(t1 + t2 + t3) / 3;
            scale_center = a5;
            scale_factor = ((norm(b6 - scale_center) / norm(a6 - scale_center)) + (norm(b7 - scale_center) / norm(a7 - scale_center))). / 2;

        elif(u1 == 1) & & (u2 == 1):
            t = -(t1 + t2) / 2;
            scale_center = a5;
            scale_factor = (norm(b6 - scale_center) / norm(a6 - scale_center));

        elif(u1 == 1) & & (u3 == 1):
            t = -(t1 + t3) / 2;
            scale_center = a5;
            scale_factor = (norm(b7 - scale_center) / norm(a7 - scale_center));

        elif(u2 == 1) & & (u3 == 1):
            t = -(t2 + t3) / 2;
            scale_center = a6;
            scale_factor = (norm(b7 - scale_center) / norm(a7 - scale_center));

        elif(u1 == 1):
            tx = mean([m1.p1(1) - s1.p1(1) m1.p2(1) - s1.p2(1) m2.p1(1) - s2.p1(1) m2.p2(1) - s2.p2(1) m3.p1(1) - s3.p1(1)
                       m3.p2(1) - s3.p2(1)]);
            ty = mean([m1.p1(2) - s1.p1(2) m1.p2(2) - s1.p2(2) m2.p1(2) - s2.p1(2) m2.p2(2) - s2.p2(2) m3.p1(2) - s3.p1(2)
                       m3.p2(2) - s3.p2(2)]);
            t = -[tx ty];
            scale_center = a5;
            scale_factor = 1;

        else:
            tx = mean([m1.p1(1) - s1.p1(1) m1.p2(1) - s1.p2(1) m2.p1(1) - s2.p1(1) m2.p2(1) - s2.p2(1) m3.p1(1) - s3.p1(1)
                       m3.p2(1) - s3.p2(1)])
            ty = mean([m1.p1(2) - s1.p1(2) m1.p2(2) - s1.p2(2) m2.p1(2) - s2.p1(2) m2.p2(2) - s2.p2(2) m3.p1(2) - s3.p1(2)
                       m3.p2(2) - s3.p2(2)])
            t = -[tx ty]
            scale_center = a5
            scale_factor = 1
'''
