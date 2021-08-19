
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    line_counts = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    line_counts = np.array(line_counts)

    times = [3.93,
    15.01,
    34.04,
    59.43,
    93.12,
    133.48,
    183.99,
    237.63,
    301.71,
    365.39]

    a = np.polyfit(line_counts, times, 2)
    print(a)

    plt.figure(1)
    plt.plot(line_counts, times, 'o')

    plt.plot(line_counts, a[0] * line_counts**2 + a[1] * line_counts + a[2], '--k')

    plt.ylabel('Dauer in s')
    plt.xlabel('Linien')

    plt.show()







