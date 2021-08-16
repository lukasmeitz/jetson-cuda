import numpy as np

from modules.components.transformation import get_transformation_cuda
from modules.handlers.load_test_sets import load_test_set
from modules.line_matcher.define_transform import transform_line_batch

if __name__ == "__main__":


    set = 70
    seed = 2000
    rng = np.random.default_rng(seed)

    # preprocess
    scene_lines, model_lines, match_id_list = load_test_set(set, "../")
    model_lines = np.array(model_lines)

    scene_lines = transform_line_batch(model_lines, 0, [50, 0], [0, 0])
    scene_lines = np.array(scene_lines)

    random_generator = np.random.default_rng(2000)
    indices = random_generator.random((200, 4))
    indices *= [len(model_lines) - 1,
                len(model_lines) - 1,
                len(scene_lines) - 1,
                len(scene_lines) - 1]
    indices = np.round(indices).astype(int)

    transformations = np.zeros((len(indices), 4))

    # evaluation
    threadsperblock = (16, 16)
    blockspergrid_x = np.math.ceil(len(indices) / threadsperblock[0])
    blockspergrid_y = np.math.ceil(len(scene_lines) / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    get_transformation_cuda[blockspergrid, threadsperblock](model_lines, scene_lines, indices, transformations)
    print(transformations)