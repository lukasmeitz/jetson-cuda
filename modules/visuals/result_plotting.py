

def compute_correct_matches(match_id_list):

    correct_count = 0

    for pair in match_id_list:

        if pair[0] == pair[1]:
            correct_count += 1

    return correct_count
