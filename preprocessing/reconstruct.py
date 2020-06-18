import cv2
import os
import numpy as np
import argparse
import sys
import collections
import torch
from shutil import copyfile, rmtree
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from reconstruction import NMFCRenderer

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_dirs(name, image_pths, args):
    id_coeffs_paths = []
    nmfc_pths = [p.replace('/images/', '/nmfcs/') for p in image_pths]
    out_paths = set(os.path.dirname(nmfc_pth) for nmfc_pth in nmfc_pths)
    for out_path in out_paths:
        mkdir(out_path)
        if args.save_cam_params:
            mkdir(out_path.replace('/nmfcs/', '/misc/'))
        if args.save_landmarks5:
            mkdir(out_path.replace('/nmfcs/', '/landmarks/'))
        if args.save_exp_params:
            mkdir(out_path.replace('/nmfcs/', '/exp_coeffs/'))
    if args.save_id_params:
        splits = set(os.path.dirname(os.path.dirname(os.path.dirname(nmfc_pth))) for nmfc_pth in nmfc_pths)
        for split in splits:
            id_coeffs_path = os.path.join(split, 'id_coeffs')
            mkdir(id_coeffs_path)
            id_coeffs_paths.append(id_coeffs_path)
    return id_coeffs_paths

def remove_images(name, image_pths):
    # Remove images (and landmarks70 if they exist)
    image_dirs_to_remove = set(os.path.dirname(image_pth) for image_pth in image_pths)
    for dir in image_dirs_to_remove:
        if os.path.isdir(dir):
            rmtree(dir)
        landmarks70_dir = dir.replace('/images/', '/landmarks70/')
        if os.path.isdir(landmarks70_dir):
            rmtree(landmarks70_dir)

def save_results(nmfcs, reconstruction_output, name, image_pths, args):
    # Create save directories
    id_coeffs_paths = make_dirs(name, image_pths, args)
    # Save
    SRT_vecs = []
    print('Saving results')
    for nmfc, cam_param, _, exp_param, landmark5, image_pth in tqdm(zip(nmfcs, *reconstruction_output, image_pths), total=len(image_pths)):
        S, R, T = cam_param
        nmfc_pth = image_pth.replace('/images/', '/nmfcs/')
        SRT_vecs.append((nmfc_pth, np.concatenate([np.array([S]), np.array(R).ravel(), np.array(T).ravel()])))
        cv2.imwrite(nmfc_pth, nmfc)
        if args.save_cam_params:
            misc_file = os.path.splitext(image_pth.replace('/images/', '/misc/'))[0] + '.txt'
            misc_file = open(misc_file, "a")
            np.savetxt(misc_file, np.array([S]))
            np.savetxt(misc_file, R)
            np.savetxt(misc_file, T)
            misc_file.close()
        if args.save_landmarks5:
            lands_file = os.path.splitext(image_pth.replace('/images/', '/landmarks/'))[0] + '.txt'
            np.savetxt(lands_file, landmark5)
        if args.save_exp_params:
            exp_params_file = os.path.splitext(image_pth.replace('/images/', '/exp_coeffs/'))[0] + '.txt'
            np.savetxt(exp_params_file, exp_param)
    if args.save_id_params:
        avg_id_params = np.mean(np.array(reconstruction_output[1]), axis=0)
        for id_coeffs_path in id_coeffs_paths:
            id_params_file = os.path.join(id_coeffs_path, name + '.txt')
            np.savetxt(id_params_file, avg_id_params)

IMG_EXTENSIONS = ['.png']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths_dict(dir):
    # Returns dict: {name: [path1, path2, ...], ...}
    image_files = {}
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname) and not '/plots' in root:
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

def dirs_exist(image_pths):
    nmfc_pths = [p.replace('/images/', '/nmfcs/') for p in image_pths]
    out_paths = set(os.path.dirname(nmfc_pth) for nmfc_pth in nmfc_pths)
    return all([os.path.exists(out_path) for out_path in out_paths])

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
    print('---------- 3D face reconstruction --------- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='head2headDataset', help='Path to the dataset directory.')
    parser.add_argument('--gpu_id', type=int, default='0', help='Negative value to use CPU, or greater equal than zero for GPU id.')
    parser.add_argument('--save_cam_params', action='store_true', default=True, help='Save the Scale, Rotation and Translation camera params for each frame.')
    parser.add_argument('--save_id_params', action='store_true', default=True, help='Save the average identity coefficient vector for each video.')
    parser.add_argument('--save_landmarks5', action='store_true', help='Save 5 facial landmarks for each frame.')
    parser.add_argument('--save_exp_params', action='store_true', default=True, help='Save the expression coefficients for each frame.')
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

    args.dataset_path = os.path.join('datasets', args.dataset_name, 'dataset')

    # Print Arguments
    print_args(parser, args)
    # Create the directory of image paths.
    images_dict = get_image_paths_dict(args.dataset_path)
    n_image_dirs = len(images_dict)
    print('Number of identities for 3D face reconstruction: %d \n' % n_image_dirs)
    # Initialize the NMFC renderer.
    renderer = NMFCRenderer(args)
    # Iterate through the images_dict
    n_completed = 0
    for name, image_pths in images_dict.items():
        n_completed += 1
        if not dirs_exist(image_pths):
            success, reconstruction_output = renderer.reconstruct(image_pths)
            if success:
                nmfcs = renderer.computeNMFCs(*reconstruction_output[:3])
                save_results(nmfcs, reconstruction_output, name, image_pths, args)
                print('(%d/%d) %s [SUCCESS]' % (n_completed, n_image_dirs, name))
            else:
                # If the 3D reconstruction not successful, remove images and video.
                remove_images(name, image_pths)
                print('(%d/%d) %s [FAILED]' % (n_completed, n_image_dirs, name))
        else:
            print('(%d/%d) %s already processed!' % (n_completed, n_image_dirs, name))
    # Clean
    renderer.clear()

if __name__=='__main__':
    main()
