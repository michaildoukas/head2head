import cv2
import os
import numpy as np
import argparse
import sys
import collections
import torch
from tqdm import tqdm

from reconstruction import NMFCRenderer

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

def compute_average_expesion_distance(expr_list1, expr_list2):
    return np.mean([np.abs(expr1 - expr2).sum() \
            for expr1, expr2 in zip(expr_list1, expr_list2)])

def main():
    print('--- Compute average L1 distance between expression coeffs --- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/head2head_finetuned_elizabeth/test_videos_latest/elizabeth/',
                        help='Path to the results directory.')
    parser.add_argument('--gpu_id', type=int, default='0', help='Negative value to use CPU, or greater equal than zero for GPU id.')
    args = parser.parse_args()
    # Figure out the device
    args.gpu_id = int(args.gpu_id)
    if args.gpu_id < 0:
        args.gpu_id = -1
    elif torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            args.gpu_id = 0
    else:
        print('GPU device not available. Exit.')
        exit(0)

    # Print Arguments
    print_args(parser, args)
    # Create the directory of image paths.
    images_dict = get_image_paths_dict(args.results_dir)
    # Make sure we have two folders, one with real and one withs fake frames.
    assert 'real' in images_dict and 'fake' in images_dict and \
           len(images_dict.keys()) == 2, 'Results directory has wrong structure'
    # Initialize the NMFC renderer.
    renderer = NMFCRenderer(args)
    # Iterate through the images_dict
    expressions_dict = {}
    for name, image_pths in images_dict.items():
        if paths_exist(image_pths):
            success, reconstruction_output = renderer.reconstruct(image_pths)
            if success:
                expressions_dict[name] = reconstruction_output[2]
            else:
                print('Reconstruction on %s failed.' % name)
                break
    # If the two expression sequences have been computed, find average L1 dist.
    if len(expressions_dict.keys()) == 2:
        dst = compute_average_expesion_distance(expressions_dict['real'],
                                                expressions_dict['fake'])
        # Average Expression Distance (AED-L1)
        print('Average expression (L1) distance between real and fake sequences (AED-L1): %0.4f' % (dst))

    # Clean
    renderer.clear()

if __name__=='__main__':
    main()
