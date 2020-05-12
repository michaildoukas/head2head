import os
import cv2
import numpy as np

def read_images(dir):
    images = []
    paths = sorted([os.path.join(dir, img_name) for img_name in os.listdir(dir)])
    for path in paths:
        images.append(cv2.imread(path))
    return images

def place_T_coordinates_on_images(images, T):
    new_images = []
    for image, t in zip(images, T):
        # Center coordinates
        center_coordinates = tuple(t.astype(np.int32))
        # Radius of circle
        radius = 4
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of 2 px
        image = cv2.circle(image, center_coordinates, radius, color, thickness)
        new_images.append(image)
    return new_images

def save_images(images, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i, image in enumerate(images):
        name = "{:06d}".format(i) + '.png'
        save_path = os.path.join(dir, name)
        cv2.imwrite(save_path, image)

def rev_nmfc(nmfcs, point):
    ret = []
    for n, nmfc in enumerate(nmfcs):
        min_dst = 99999
        arg_min = None
        if n == 0:
            lim_i_l, lim_i_h = 0, 255
            lim_j_l, lim_j_h = 0, 255
        else:
            lim_i_l, lim_i_h = prev_arg_min[0]-20, prev_arg_min[0]+20
            lim_j_l, lim_j_h = prev_arg_min[1]-20, prev_arg_min[1]+20
        for i in range(lim_i_l, lim_i_h):
            for j in range(lim_j_l, lim_j_h):
                dst = sum(abs(nmfc[i,j,:] - point))
                if dst < min_dst:
                    min_dst = dst
                    arg_min = (i, j)
        prev_arg_min = arg_min
        a = np.flip(np.array(list(arg_min)))
        ret.append(a)
    return ret

# BGR eye values in nmfc images.
left_eye_nmfc = np.array([192, 180, 81])
right_eye_nmfc = np.array([192, 180, 171])

lst = ['Obama', 'Merkel', 'May']

for l in lst:
    print(l)
    images_dir = 'datasets/head2headDataset/dataset/test/images/' + l
    nmfcs_dir = 'datasets/head2headDataset/dataset/test/nmfcs/' + l
    save_dir = 'temp/' + l
    images = read_images(images_dir)
    nmfcs = read_images(nmfcs_dir)
    cords = rev_nmfc(nmfcs, point)
    for nmfc, cord in zip(nmfcs, cords):
        print(cord, nmfc[cord[1], cord[0]])
    new_images = place_T_coordinates_on_images(images, cords)
    save_images(new_images, save_dir)
