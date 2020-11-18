import cv2
import os
import numpy as np
import argparse
import collections
from skimage import io
import torch
import itertools
import dlib
from tqdm import tqdm
import util.util as util
from detect_landmarks70 import detect_landmarks
from reconstruction import _procrustes

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

def compute_eye_landmarks_distance(e1, e2):
    error = abs(e1 - e2)
    dsts = np.linalg.norm(error, axis=1)
    return np.mean(dsts)

def main():
    print('Computation of average eye landmarks L2-distance\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/head2head_obama/latest_epoch/videos_test/obama',
                        help='Path to the results directory.')
    parser.add_argument('--do_alignment', action='store_true', help='Align eye landmarks with procrustes.')
    args = parser.parse_args()

    predictor_path = 'preprocessing/files/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Print Arguments
    print_args(parser, args)
    # Create the directory of image paths.
    images_dict = get_image_paths_dict(args.results_dir)
    # Make sure we have two folders, one with real and one withs fake frames.
    assert 'real' in images_dict and 'fake' in images_dict and \
           len(images_dict.keys()) == 2, 'Results directory has wrong structure'

    f_landmarks = detect_landmarks(images_dict['fake'], detector, predictor)
    r_landmarks = detect_landmarks(images_dict['real'], detector, predictor)
    distances = []
    for f_land, r_land in zip(f_landmarks, r_landmarks):
        f_land_eyes = np.concatenate([f_land[36:48, :], f_land[68:70, :]], axis=0)
        r_land_eyes = np.concatenate([r_land[36:48, :], r_land[68:70, :]], axis=0)
        if args.do_alignment:
            _, f_land_eyes, _ = _procrustes(r_land_eyes, f_land_eyes)
        distances.append(compute_eye_landmarks_distance(f_land_eyes, r_land_eyes))
    print('Average eye landmarks L2-distance: %0.4f' % np.mean(distances))

if __name__=='__main__':
    main()
