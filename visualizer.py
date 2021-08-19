from modules.handlers.load_result_files import collect_file_list, read_dict_data
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    files = collect_file_list("results_final_backup")
    data = read_dict_data(files)

    print("read " + str(len(data)) + " test set results")

    # first algorithm data
    runtimes_first = {}
    runtimes_standard = {}
    runtimes_cuda = {}

    test_cases = [2, 50, 5, 10, 12, 22, 24, 25, 37, 43, 51, 53, 62, 67, 69]
    test_cases_range = np.arange(len(test_cases))

    for test_case in data:

        # sum runtimes per set
        if test_case["ransac_type"] == "first":

            if test_case["test_set_number"] in runtimes_first:
                runtimes_first[test_case["test_set_number"]] += test_case["duration"]
            else:
                runtimes_first[test_case["test_set_number"]] = test_case["duration"]

        # sum runtimes per set
        if test_case["ransac_type"] == "standard":

            if test_case["test_set_number"] in runtimes_standard:
                runtimes_standard[test_case["test_set_number"]] += test_case["duration"]
            else:
                runtimes_standard[test_case["test_set_number"]] = test_case["duration"]

        # sum runtimes per set
        if test_case["ransac_type"] == "final":

            if test_case["test_set_number"] in runtimes_cuda:
                runtimes_cuda[test_case["test_set_number"]] += test_case["duration"]
            else:
                runtimes_cuda[test_case["test_set_number"]] = test_case["duration"]

    runtimes_first_plottable = []
    for key in test_cases:
        runtimes_first[key] /= 15
        runtimes_first_plottable.append(runtimes_first[key])

    runtimes_standard_plottable = []
    for key in test_cases:
        runtimes_standard[key] /= 15
        runtimes_standard_plottable.append(runtimes_standard[key])

    runtimes_cuda_plottable = []
    for key in test_cases:
        runtimes_cuda[key] /= 15
        runtimes_cuda_plottable.append(runtimes_cuda[key])


    print(test_cases)
    print(runtimes_standard_plottable)

    plt.figure(1)
    plt.plot(test_cases_range + 0.0,  runtimes_first_plottable, color='r')
    plt.ylabel('Dauer in s')
    plt.xlabel('Testfall')


    plt.figure(1)
    plt.plot(test_cases_range + 0.25,  runtimes_standard_plottable, color='g')
    plt.ylabel('Dauer in s')
    plt.xlabel('Testfall')


    plt.figure(1)
    plt.plot(test_cases_range + 0.5,  runtimes_cuda_plottable, color='b')
    plt.ylabel('Dauer in s')
    plt.xlabel('Testfall')

    plt.show()
