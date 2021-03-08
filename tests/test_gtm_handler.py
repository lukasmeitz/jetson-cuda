
from modules.handlers.gtm_handler import read_gtm_file
from modules.visuals.imaging import load_image, save_image, draw_circles, concat, draw_lines

def test_gtm(path):

    # parameters
    img_left = 'img1'
    img_right = 'img2'
    ratio = '20'

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

    # save separate images with highlighted features
    save_image(image_1, 'results/img1_manipulated.png')
    save_image(image_2, 'results/img1_manipulated_2.png')

    # concat the images
    image_concat = concat(image_1, image_2)
    height, width, channels = image_1.shape
    match_lines = []


    # create line vectors
    for match in matches:

        i_l = int(match[0])
        i_r = int(match[1])

        x1 = float(key_points_left[i_l][0])
        y1 = float(key_points_left[i_l][1])
        x2 = width + float(key_points_right[i_r][0])
        y2 = float(key_points_right[i_r][1])

        match_lines += [[x1, y1, x2, y2]]

    # draw lines to combine matches
    draw_lines(image_concat, match_lines, color=(0, 0, 255))

    # save concatenated image
    save_image(image_concat, 'results/image_concat.png')

    # done
    return
