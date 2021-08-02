
import scipy.io


if __name__ == "__main__":

    data = scipy.io.loadmat('../data/camera/K_synthetic.mat')
    print(data['K'])
