from modules.algorithms.preprocessing import filter_lines
from modules.components.preprocessor import preprocessor, preprocess_length
import numpy as np

from modules.handlers.load_test_sets import load_test_set
from modules.visuals.imaging import draw_lines, plot_image, load_image

if __name__ == "__main__":

    scene_lines, model_lines, match_id_list = load_test_set(1, '../', demo=True)
    set_num = "{:03d}".format(1)
    image = load_image('../data/TestSets_Demonstrator/TestSet' + set_num + '/image.png')


    # load
    scene_lines = np.array(scene_lines)

    # stage 1
    scene_lines_filtered = filter_lines(scene_lines, max_lines=80)

    # stage 2
    scene_line_pairs = preprocessor(scene_lines, max_lines=1000, debug=True)
    scene_lines_preprocessed_indices = []
    for lp in scene_line_pairs:
        if int(lp[0]) in scene_lines_preprocessed_indices:
            pass
        else:
            scene_lines_preprocessed_indices.append(int(lp[0]))

        if int(lp[1]) in scene_lines_preprocessed_indices:
            pass
        else:
            scene_lines_preprocessed_indices.append(int(lp[1]))
    scene_lines_preprocessed = scene_lines[scene_lines_preprocessed_indices]


    # stage 3
    scene_line_pairs_lengthsort = preprocess_length(scene_lines, max_lines=160, debug=True)

    # visualize
    draw_lines(image, scene_lines, color=(0, 0, 255))
    draw_lines(image, scene_lines_filtered, color=(255, 255, 0))
    draw_lines(image, scene_line_pairs_lengthsort, color=(0, 255, 255))
    draw_lines(image, scene_lines_preprocessed, color=(255, 0, 255))
    plot_image(image, "Current Iteration", blocking=True)
