import cv2
import os
import numpy as np
import argparse
import collections
import torch
import itertools
from tqdm import tqdm
from preprocessing import transform
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

def l1_dist(v1, v2):
    return np.abs(v1 - v2).sum()

def euler_dist(e1, e2):
    d0 = abs(e1[0]-e2[0])
    if d0 > 180:
        d0 = 360 - d0
    d1 = abs(e1[1]-e2[1])
    if d1 > 180:
        d1 = 360 - d1
    d2 = abs(e1[2]-e2[2])
    if d2 > 180:
        d2 = 360 - d2
    return (d0 + d1 + d2) / 3

def get_within_distances(lst):
    pairs = itertools.combinations(lst, 2)
    max = 0
    min = np.float('inf')
    avg = []
    for pair in pairs:
        dst = l1_dist(pair[0], pair[1])
        if dst < min:
            min = dst
        if dst > max:
            max = dst
        avg.append(dst)
    avg = np.mean(avg)
    return min, max, avg

def compute_distance_of_average_identities(ident_list1, ident_list2):
    avg_ident1, avg_ident2 = np.mean(ident_list1, axis=0), np.mean(ident_list2, axis=0)
    return l1_dist(avg_ident1, avg_ident2)

def compute_average_expesion_distance(expr_list1, expr_list2):
    return np.mean([l1_dist(expr1, expr2) \
            for expr1, expr2 in zip(expr_list1, expr_list2)])

def compute_average_rotation_distance(cam_list1, cam_list2):
    # Rotation parameters to Euler angles.
    angles_list1 = [transform.matrix2angle(cam[1]) for cam in cam_list1]
    angles_list2 = [transform.matrix2angle(cam[1]) for cam in cam_list2]
    return np.mean([euler_dist(ang1, ang2) \
            for ang1, ang2 in zip(angles_list1, angles_list2)])

def main():
    print('Computation of L1 distance between average identity coeffs (DAI-L1)\n')
    print('Computation of average L1 distance between expression coeffs (AED-L1)\n')
    print('Computation of average L1 distance between rotation parameters (ARD-L1)\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/head2head_obama/latest_epoch/videos_test/obama',
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
    identities_dict = {}
    expressions_dict = {}
    camera_dict = {}
    for name, image_pths in images_dict.items():
        if paths_exist(image_pths):
            success, reconstruction_output = renderer.reconstruct(image_pths)
            if success:
                identities_dict[name] = reconstruction_output[1]
                expressions_dict[name] = reconstruction_output[2]
                camera_dict[name] = reconstruction_output[0]
            else:
                print('Reconstruction on %s failed.' % name)
                break
    # If the two expression sequences have been computed, find average L1 dist.
    if len(identities_dict.keys()) == 2:
        # Identity
        dai_L1 = compute_distance_of_average_identities(identities_dict['real'],
                                                        identities_dict['fake'])
        # Distance Between Average Identities (DAI-L1)
        print('(L1) distance between average identities from real and fake sequences (DAI-L1): %0.4f' % (dai_L1))
        #dsts_real = get_within_distances(identities_dict['real'])
        #print('Within real sequence min %0.4f, max %0.4f, mean %0.4f' % dsts_real)
        #dsts_fake = get_within_distances(identities_dict['fake'])
        #print('Within fake sequence min %0.4f, max %0.4f, mean %0.4f' % dsts_fake)
        # Expression
        aed_L1 = compute_average_expesion_distance(expressions_dict['real'],
                                                   expressions_dict['fake'])
        # Average Expression Distance (AED-L1)
        print('Average expression (L1) distance between real and fake sequences (AED-L1): %0.4f' % (aed_L1))
        # Pose
        ard_L1 = compute_average_rotation_distance(camera_dict['real'],
                                                   camera_dict['fake'])
        # Average Rotation Parameters Distance (ARD-L1)
        print('Average rotation (L1) distance between real and fake sequences (ARD-L1): %0.4f' % (ard_L1))

    # Clean
    renderer.clear()

if __name__=='__main__':
    main()
