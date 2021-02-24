
from modules.handlers.gtm_handler import read_gtm_file
from modules.visuals.imaging import load_image, save_image, draw_circles

def test_gtm(path):

    img_left = 'img1'
    img_right = 'img2'

    ratio = '100'

    # read a gtm file
    key_points_left, key_points_right, matches, inliers = read_gtm_file(
            path + 'gtm_data/SIFT/' + img_left + '_' + img_right + '_inlRat' + ratio + '0SIFT.gtm')
    print(matches)

    # read the corresponding images
    image_1 = load_image( path + 'gtm_data/images/' + img_left + '.png')
    image_2 = load_image( path + 'gtm_data/images/' + img_right + '.png')

    # plot the fuck out of the data
    # draw match points in first image
    match_points = []
    for kpl in key_points_left:
        match_points += [[float(kpl[0]), float(kpl[1]), 5]]
    draw_circles(image_1, match_points, color=(0, 255, 0))

    # draw match points in second image
    match_points = []
    for kpr in key_points_right:
        match_points += [[float(kpr[0]), float(kpr[1]), 5]]
    draw_circles(image_2, match_points, color=(0, 255, 0))

    # save first image
    save_image(image_1, 'results/img1_manipulated.png')
    save_image(image_2, 'results/img1_manipulated_2.png')

    # done
    return