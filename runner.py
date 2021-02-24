from sys import platform

from modules.handlers import *
from modules.visuals.imaging import *
from tests.test_imaging import *
from tests.test_gtm_handler import *

if __name__ == "__main__":



    if platform == "linux" or platform == "linux2":
        path = "home/lukas/jetson-cuda/data/"
    else:
        path = "data/"


    test_gtm(path)