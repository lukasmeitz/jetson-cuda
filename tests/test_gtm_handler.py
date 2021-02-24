
from modules.handlers.gtm_handler import read_gtm_file
from modules.visuals.imaging import load_image, save_image, draw_circles

def test_gtm():

    # read a gtm file
    key_points_left, key_points_right, matches, inliers = read_gtm_file('/home/lukas/jetson-cuda/data/gtm_data/SIFT/img1_img2_inlRat1000SIFT.gtm')
    print(matches)


    # read the corresponding images
    image_1 = load_image('/home/lukas/jetson-cuda/data/gtm_data/images/img1.png')
    image_2 = load_image('/home/lukas/jetson-cuda/data/gtm_data/images/img2.png')

    # plot the fuck out of the data
    # draw match points in first image
    match_points = []
    for kpl in key_points_left:
        match_points += [float(kpl[0]), float(kpl[1]), 5]

    draw_circles(image_1, match_points)

    # save first image
    save_image(image_1, '/home/lukas/jetson-cuda/data/gtm_data/images/img1_manipulated.png')

    # done
    return