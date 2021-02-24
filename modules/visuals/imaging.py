
import cv2
import numpy as np


def load_image(path):
    image = cv2.imread(path)
    print('image read sucessfully')
    return image


def save_image(image, path):
    cv2.imwrite(path, image)


def plot_image(image, blocking=False):

    # draw using cv2 library
    cv2.imshow('image', image)

    if blocking:
        cv2.waitKey()


# format: np.array([x1,y1, x2, y2], ...)
def draw_lines(image, lines, color=(128, 128, 128)):

    for line in lines:
        cv2.line(image,
                 (round(line[0]), round(line[1])),
                 (round(line[2]), round(line[3])),
                 color,
                 thickness=2)


# format: np.array([x, y], ...)
def draw_circles(image, circles, color=(128, 128, 128)):

    for circle in circles:
        cv2.circle(image,
                 (round(circle[0]), round(circle[1])),
                 circle[2],
                 color,
                 thickness=1)