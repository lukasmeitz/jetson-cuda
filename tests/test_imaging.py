
import visuals

image = visuals.load_image('../data/gtm_data/images/img1.png')
visuals.draw_lines(image, [[0, 0, 300, 300]], color=(255, 0, 0))
visuals.save_image(image, '../results/image_out.png')