from modules.handlers.load_result_files import collect_file_list, read_dict_data
import matplotlib.pyplot as plt

from modules.visuals.result_plotting import compute_correct_matches

if __name__ == "__main__":

    files = collect_file_list("results")
    data = read_dict_data(files)

    print("read " + str(len(data)) + " test set results")

    # separate algorithms
    adaptive_data = [[], []]
    standard_data = [[], []]
    randomised_data = [[], []]
    possible_matches_gtm = []



    for test_case in data:

        if test_case["ransac_type"] == "adaptive":
            adaptive_data[0] += [test_case["duration"]]
            adaptive_data[1] += [compute_correct_matches(test_case["match_id_list"])]
            possible_matches_gtm += [test_case["match_count_gtm"]]

        if test_case["ransac_type"] == "standard":
            standard_data[0] += [min([test_case["duration"], 1000])]
            standard_data[1] += [compute_correct_matches(test_case["match_id_list"])]

        if test_case["ransac_type"] == "randomised":
            randomised_data[0] += [test_case["duration"]]
            randomised_data[1] += [compute_correct_matches(test_case["match_id_list"])]

    plt.figure(1)
    plt.stem(range(len(adaptive_data[0])), adaptive_data[0], 'r')
    plt.stem(range(len(standard_data[0])), standard_data[0], 'b')
    plt.stem(range(len(randomised_data[0])), randomised_data[0], 'g')
    plt.ylabel('duration in s')
    plt.xlabel('test case number')

    plt.figure(2)
    plt.plot(possible_matches_gtm, 'k-')
    plt.plot(adaptive_data[1], 'r')
    plt.plot(standard_data[1], 'b')
    plt.plot(randomised_data[1], 'g')
    plt.ylabel('correct matches')
    plt.xlabel('test case number')
    plt.show()


    print(adaptive_data)
