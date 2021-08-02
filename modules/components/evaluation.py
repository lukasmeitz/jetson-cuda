
import numpy as np


# Kreuzdistanz Linienpaare
def calc_cross_distance_pairs(ml1, ml2, sl1, sl2):

    dist = 0.0

    # ml1 sl1
    dist += calc_cross_distance(ml1, sl1)
    # ml1 sl2
    dist += calc_cross_distance(ml1, sl2)
    # ml2 sl1
    dist += calc_cross_distance(ml2, sl1)
    # ml2 sl2
    dist += calc_cross_distance(ml2, sl2)

    return dist / 4


# Kreuzdistanz Linien
def calc_cross_distance(ml, sl):

    len1 = calc_distance(ml[0:2], ml[2:4])
    len2 = calc_distance(sl[0:2], sl[2:4])
    len_avg = (len1+len2) / 2

    dist = 0.0

    # mlp1 slp1
    dist += calc_distance(ml[0:2], sl[0:2])
    # mlp2 slp1
    dist += calc_distance(ml[2:4], sl[0:2])
    # mlp1 slp2
    dist += calc_distance(ml[0:2], sl[2:4])
    # mlp2 slp2
    dist += calc_distance(ml[2:4], sl[2:4])

    return dist / 4


def calc_advanced_cross_distance(ml, sl):

    len1 = calc_distance(ml[0:2], ml[2:4])
    len2 = calc_distance(sl[0:2], sl[2:4])
    len_avg = (len1+len2) / 2

    dist = []

    # mlp1 slp1
    dist += [calc_distance(ml[0:2], sl[0:2])]
    # mlp2 slp1
    dist += [calc_distance(ml[2:4], sl[0:2])]
    # mlp1 slp2
    dist += [calc_distance(ml[0:2], sl[2:4])]
    # mlp2 slp2
    dist += [calc_distance(ml[2:4], sl[2:4])]
    #midpoints
    dist += []

    return np.min(dist)



# Minimale Distanze Linienpaare
def calc_min_pair_distance(ml1, ml2, sl1, sl2):

    error = 999

    # distance of all points to one another
    dist_11 = calc_min_distance(ml1, sl1)
    dist_12 = calc_min_distance(ml1, sl2)

    if dist_11 < dist_12:
        dist_2 = calc_min_distance(ml2, sl2)
        error = dist_11 + dist_2
    else:
        dist_2 = calc_min_distance(ml2, sl1)
        error = dist_12 + dist_2

    return error / 2


# Minimale Distanz
def calc_min_distance(line1, line2):

    # line1 point 1 to line2 distance
    dist_11 = calc_distance(line1[0:2], line2[0:2])
    dist_12 = calc_distance(line1[0:2], line2[2:4])

    if dist_11 < dist_12:
        dist_2 = calc_distance(line1[2:4], line2[2:4])
        error = dist_11 + dist_2
    else:
        dist_2 = calc_distance(line1[2:4], line2[0:2])
        error = dist_12 + dist_2

    return error / 2


# Attribute für Linienpaare
def calc_attribute_error_pairs(ml1, ml2, sl1, sl2):

    error_11 = calc_attribute_error(ml1, sl1)
    error_12 = calc_attribute_error(ml1, sl2)

    if error_11 < error_12:
        error = error_11 + calc_attribute_error(ml2, sl2)
    else:
        error = error_12 + calc_attribute_error(ml2, sl1)

    return error / 2


# Attribute für Linien
def calc_attribute_error(ml, sl):

    # length
    len1 = calc_distance(ml[0:2], ml[2:4])
    len2 = calc_distance(sl[0:2], sl[2:4])
    len_diff = np.abs((len1-len2))
    len_norm = len_diff / ((len1+len2)/2)

    # angle
    angle = calc_angle(ml, sl)
    angle_norm = angle / 90

    # distance
    dist = calc_min_distance(ml, sl)
    dist_norm = dist / 50

    # distance midpoint

    return dist_norm + angle_norm + len_norm



# Winkel zwischen zwei Vektoren
def calc_angle(a, b):

    cos_winkel_1 = a[0] * b[0] + a[1] * b[1]
    cos_winkel_2 = -a[1] * b[0] + a[0] * b[1]
    winkel_radians = np.arccos(min(abs(cos_winkel_1), 1))
    winkel_grad = np.degrees(winkel_radians)

    if cos_winkel_1 * cos_winkel_2 < 0:
        winkel_grad = -winkel_grad

    return winkel_grad


# Maximale Distanz Linienpaare
def calc_max_pair_distance(ml1, ml2, sl1, sl2):

    err_11 = calc_min_distance(ml1, sl1)
    err_12 = calc_min_distance(ml1, sl2)

    if err_11 < err_12:
        error = np.max([calc_max_distance(ml1, sl1),
                        calc_max_distance(ml2, sl2)])
    else:
        error = np.max([calc_max_distance(ml1, sl2),
                        calc_max_distance(ml2, sl1)])

    return error

# Maximale Distanz
def calc_max_distance(line1, line2):

    dist_11 = calc_distance(line1[0:2], line2[0:2])
    dist_12 = calc_distance(line1[0:2], line2[2:4])

    if dist_11 < dist_12:
        error = np.max([dist_11, calc_distance(line1[2:4], line2[2:4])])
    else:
        error = np.max([dist_12, calc_distance(line1[2:4], line2[0:2])])

    return error


# Euklidische Distanz
def calc_distance(p1, p2):

    x_dist = p2[0] - p1[0]
    y_dist = p2[1] - p1[1]
    dist = np.sqrt(x_dist ** 2 + y_dist ** 2)

    return dist



