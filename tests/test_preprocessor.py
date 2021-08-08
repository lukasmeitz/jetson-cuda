from modules.components.preprocessor import preprocessor
import numpy as np

from modules.handlers.load_test_sets import load_test_set

if __name__ == "__main__":

    scene_lines, model_lines, match_id_list = load_test_set(70, "../")


    model_lines = np.array(model_lines)
    scene_lines = np.array(scene_lines)

    scene_line_pairs = preprocessor(scene_lines, max_lines=120)
    model_line_pairs = preprocessor(model_lines, max_lines=120)