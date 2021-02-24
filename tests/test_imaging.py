from modules.visuals.imaging import *

def test_imaging():

    image = load_image('/home/lukas/jetson-cuda/data/gtm_data/images/img1.png')
    draw_lines(image, [[0, 0, 300, 300]], color=(255, 0, 0))
    save_image(image, '/home/lukas/jetson-cuda/results/image_out.png')