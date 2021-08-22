from modules.handlers.load_test_sets import load_test_set
import numpy as np

from modules.visuals.imaging import draw_lines, plot_image

if __name__ == "__main__":

    # prepare
    center = np.array([1280/2, 720/2]) # np.array([256, 256]) #
    set = 55

    # preprocess
    scene_lines, model_lines, match_id_list = load_test_set(set, "../")
    model_lines = np.array(model_lines)
    scene_lines = np.array(scene_lines)


    image_size = center * 2
    blank_image = np.ones((int(image_size[1]), int(image_size[0]), 3), np.uint8) * 255

    # draw scene lines
    draw_lines(blank_image, scene_lines, (0, 0, 255))
    plot_image(blank_image, "test set " + str(set), True)