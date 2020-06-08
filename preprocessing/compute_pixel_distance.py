import cv2
import os
import numpy as np
import argparse
import collections
import torch
import itertools
from tqdm import tqdm
import util.util as util

IMG_EXTENSIONS = ['.png']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths_dict(dir):
    # Returns dict: {name: [path1, path2, ...], ...}
    image_files = {}
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        basename = os.path.basename(root)
        for fname in fnames:
            if is_image_file(fname) and basename in ['real', 'fake']:
                path = os.path.join(root, fname)
                seq_name = os.path.basename(root).split('_')[0]
                if seq_name not in image_files:
                    image_files[seq_name] = [path]
                else:
                    image_files[seq_name].append(path)
    # Sort paths for each sequence
    for k, v in image_files.items():
        image_files[k] = sorted(v)
    # Return directory sorted for keys (identity names)
    return collections.OrderedDict(sorted(image_files.items()))

def paths_exist(image_pths):
    return all([os.path.exists(image_path) for image_path in image_pths])

def print_args(parser, args):
    message = ''
    message += '----------------- Arguments ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '-------------------------------------------'
    print(message)

def main():
    print('Computation of average pixel distance (APD)\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/head2head_obama/latest_epoch/videos_test/obama',
                        help='Path to the results directory.')
    args = parser.parse_args()

    # Print Arguments
    print_args(parser, args)
    # Create the directory of image paths.
    images_dict = get_image_paths_dict(args.results_dir)
    # Make sure we have two folders, one with real and one withs fake frames.
    assert 'real' in images_dict and 'fake' in images_dict and \
           len(images_dict.keys()) == 2, 'Results directory has wrong structure'

    total_distance, total_pixels = 0, 0
    for f_path, r_path in zip(images_dict['fake'], images_dict['real']):
        f_img = cv2.imread(f_path)
        r_img = cv2.imread(r_path)
        total_distance, total_pixels, _ = util.get_pixel_distance(
                r_img, f_img, total_distance, total_pixels)
    print('Average pixel (L2) distance for sequence (APD-L2): %0.2f' % (total_distance/total_pixels))

if __name__=='__main__':
    main()
