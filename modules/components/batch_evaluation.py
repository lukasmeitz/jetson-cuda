from modules.components.evaluation import calc_min_distance


def count_inlier_standard(model_lines, scene_lines, threshold=35):

    inlier = 0
    matches = []
    for m in range(len(model_lines)):
        for s in range(len(scene_lines)):

            if calc_min_distance(model_lines[m], scene_lines[s]) < threshold:
                inlier += 1
                matches.append([model_lines[m][7], scene_lines[s][7]])

    return inlier, matches


def count_inlier_mle(model_lines, scene_lines, threshold=35):
    inlier = 0
    error = 0.0
    matches = []
    for m in range(len(model_lines)):
        for s in range(len(scene_lines)):

            tmp_error = calc_min_distance(model_lines[m], scene_lines[s])
            if tmp_error < threshold:
                inlier += 1
                error += tmp_error
                matches.append([model_lines[m][7], scene_lines[s][7]])
            else:
                error += threshold

    return inlier, matches, error