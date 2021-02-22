import time

import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import sys

from load_lines_data import load_lines_data
from define_transform import define_transformation, transform_modelline_batch


def ransac_plot(n, x, y, m, c, final=False, x_in=(), y_in=(), points=()):
    """ plot the current RANSAC step
    :param n      iteration
    :param points picked up points for modeling
    :param x      samples x
    :param y      samples y
    :param m      slope of the line model
    :param c      shift of the line model
    :param x_in   inliers x
    :param y_in   inliers y
    """

    fname = "output/figure_" + str(n) + ".png"
    line_width = 1.
    line_color = '#0080ff'
    title = 'iteration ' + str(n)

    if final:
        fname = "output/final.png"
        line_width = 3.
        line_color = '#ff0000'
        title = 'final solution'

    # grid for the plot
    grid = [min(x) - 10, max(x) + 10, min(y) - 20, max(y) + 20]
    plt.axis(grid)
    #print(grid)

    # put grid on the plot
    plt.grid(b=True, which='major', color='0.75', linestyle='--')

    plt.xticks([i for i in range(round(min(x)[0]) - 10, round(max(x)[0]) + 10, 5)])
    plt.yticks([i for i in range(round(min(y)[0]) - 20, round(max(y)[0]) + 20, 10)])

    # plot input points
    plt.plot(x[:,0], y[:,0], marker='o', label='Input points', color='#00cc00', linestyle='None', alpha=0.4)

    # draw the current model
    plt.plot(x, m*x + c, 'r', label='Line model', color=line_color, linewidth=line_width)

    # draw inliers
    if not final:
        plt.plot(x_in, y_in, marker='o', label='Inliers', linestyle='None', color='#ff0000', alpha=0.6)

    # draw points picked up for the modeling
    if not final:
        plt.plot(points[:,0], points[:,1], marker='o', label='Picked points', color='#0000cc', linestyle='None', alpha=0.6)

    plt.title(title)
    plt.legend()
    #plt.savefig(fname)
    plt.draw()
    plt.pause(0.5)
    #plt.close()
    plt.clf()


# Ransac parameters
ransac_iterations = 15  # number of iterations
ransac_threshold = 14   # threshold

center_point = [360, 640]

# ransac base line data
model_lines, scene_lines = load_lines_data()
current_best_inliers = 0

# perform RANSAC iterations
for it in range(ransac_iterations):

    # pick a random pair
    random_sample_index = round(np.random.random() * len(model_lines))

    sample_model_line_pair = model_lines[random_sample_index]
    sample_scene_line_pair = scene_lines[random_sample_index]

    # find a line model for these points
    t = define_transformation([sample_model_line_pair[0], sample_model_line_pair[1],
                               sample_scene_line_pair[0], sample_scene_line_pair[1]],
                              center_point)

    # convert all other model lines to this transformation
    model_lines_transformed = transform_modelline_batch(model_lines, t[0], [t[1], t[2]], center_point)

    # find inliers
    inlier_index_list = []
    num_inliers = 0

    for ind in range(len(model_lines)):

        # take model and scene lines
        ml1, ml2 = model_lines_transformed[ind]
        sl1, sl2 = scene_lines[ind]

        # calculate an error measure
        MS11 = np.sqrt((ml1[0] - sl1[0]) ** 2 + (ml1[1] - sl1[1]) ** 2)
        MS12 = np.sqrt((ml1[0] - sl2[0]) ** 2 + (ml1[1] - sl2[1]) ** 2)
        MS21 = np.sqrt((ml2[0] - sl1[0]) ** 2 + (ml2[1] - sl1[1]) ** 2)
        MS22 = np.sqrt((ml2[0] - sl2[0]) ** 2 + (ml2[1] - sl2[1]) ** 2)

        error = (MS11 + MS12 + MS21 + MS22) / 4

        # check whether it's an inlier or not
        if error < ransac_threshold:
            inlier_index_list += [ind]
            num_inliers += 1

    # in case a new model is better - cache it
    if num_inliers > current_best_inliers:
        current_best_transformation = t
        current_best_inliers = num_inliers
        current_best_inliers_indices = inlier_index_list

    print("inliers: " + str(num_inliers))

# finish this
print('\nFinal transformation:\n')
print('  inliers = ', current_best_inliers)
print('  transformation = ', current_best_transformation)


# plot the final model
# ransac_plot(0, x_noise,y_noise, model_m, model_c, True)

# pics or it didn´t happen
image_data = scipy.io.loadmat('../data_in/img.mat')
image_data = np.array(image_data["img"])
image = np.ones((len(image_data), len(image_data[0]))) * 255

# make sure the right transformation is calculated
model_lines_transformed = transform_modelline_batch(model_lines, current_best_transformation[0], [current_best_transformation[1], current_best_transformation[2]], center_point)

# draw model lines
for line_pair_index in current_best_inliers_indices:

    # resolve index
    ml1_p1 = model_lines_transformed[line_pair_index][0][0:2]
    ml1_p2 = model_lines_transformed[line_pair_index][0][2:4]

    ml2_p1 = model_lines_transformed[line_pair_index][1][0:2]
    ml2_p2 = model_lines_transformed[line_pair_index][1][2:4]

    sl1_p1 = scene_lines[line_pair_index][0][0:2]
    sl1_p2 = scene_lines[line_pair_index][0][2:4]

    sl2_p1 = scene_lines[line_pair_index][1][0:2]
    sl2_p2 = scene_lines[line_pair_index][1][2:4]

    # draw model lines
    cv2.line(image, (round(ml1_p1[0]), round(ml1_p1[1])), (round(ml1_p2[0]), round(ml1_p2[1])), (0, 0, 255), thickness=2)
    cv2.line(image, (round(ml2_p1[0]), round(ml2_p1[1])), (round(ml2_p2[0]), round(ml2_p2[1])), (0, 0, 255), thickness=2)

    # draw scene lines thinner
    cv2.line(image, (round(sl1_p1[0]), round(sl1_p1[1])), (round(sl1_p2[0]), round(sl1_p2[1])), (0, 0, 255), thickness=1)
    cv2.line(image, (round(sl2_p1[0]), round(sl2_p1[1])), (round(sl2_p2[0]), round(sl2_p2[1])), (0, 0, 255), thickness=1)

cv2.imshow('image', image)
cv2.waitKey()

