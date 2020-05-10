import cv2
import os
import numpy as np
import torch
import argparse
import sys
import scipy.io as io
from shutil import copyfile

from reconstruction import NMFCRenderer

def mkdirs(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def save_results(source_nmfcs, source_images_paths, args):
    assert len(source_nmfcs) == len(source_images_paths), \
            'Rendered NMFC and original source sequence have different lengths.'
    save_nmfcs_dir = os.path.join(args.dataset_path, args.split_t,
                        'source_nmfcs', args.target_id + '_' + args.source_id)
    save_images_dir = os.path.join(args.dataset_path, args.split_t,
                        'source_images', args.target_id + '_' + args.source_id)
    mkdirs([save_nmfcs_dir, save_images_dir])
    for i, source_images_path in enumerate(source_images_paths):
        frame_name = os.path.basename(source_images_path)
        copyfile(source_images_path, os.path.join(save_images_dir, frame_name))
        cv2.imwrite(os.path.join(save_nmfcs_dir, frame_name), source_nmfcs[i])

def compute_cam_params(s_cam_params, t_cam_params, args):
    cam_params = s_cam_params
    if args.adapt_scale:
        mean_S_target = np.mean([params[0] for params in t_cam_params])
        mean_S_source = np.mean([params[0] for params in s_cam_params])
        if args.standardize:
            std_S_target = np.std([params[0] for params in t_cam_params])
            std_S_source = np.std([params[0] for params in s_cam_params])
            S = [(params[0] - mean_S_source) * std_S_target / std_S_source \
                 + mean_S_target for params in s_cam_params]
        else:
            S = [params[0] - mean_S_source + mean_S_target
                 for params in s_cam_params]
        # Normalised Translation for source and target.
        nT_target = [params[2] / params[0] for params in t_cam_params]
        nT_source = [params[2] / params[0] for params in s_cam_params]
        cam_params = [(s, params[1], s * t) \
                      for s, params, t in zip(S, s_cam_params, nT_source)]
        if args.adapt_scale_and_translation:
            mean_nT_target = np.mean(nT_target, axis=0)
            mean_nT_source = np.mean(nT_source, axis=0)
            if args.standardize:
                std_nT_target = np.std(nT_target, axis=0)
                std_nT_source = np.std(nT_source, axis=0)
                nT = [(t - mean_nT_source) * std_nT_target / std_nT_source \
                     + mean_nT_target for t in nT_source]
            else:
                nT = [t - mean_nT_source + mean_nT_target
                      for t in nT_source]
            cam_params = [(s, params[1], s * t) \
                          for s, params, t in zip(S, s_cam_params, nT)]
    return cam_params

def read_params(params_type, path, speaker_id):
    if params_type  == 'id':
        path = os.path.join(path, speaker_id + '.txt')
        if os.path.exists(path):
            return np.loadtxt(path), None
    if params_type == 'exp' or params_type == 'cam':
        txt_files = []
        params = []
        parts = os.listdir(path)
        base_part = os.path.join(path, speaker_id)
        for part in sorted(parts):
            dir = os.path.join(path, part)
            if base_part in dir:
                txt_files.extend([os.path.join(dir, txt) \
                                 for txt in sorted(os.listdir(dir))])
        for f in txt_files:
            if os.path.exists(f):
                if params_type == 'exp':
                    params.append(np.loadtxt(f))
                else:
                    S = np.loadtxt(f, max_rows=1)
                    R = np.loadtxt(f, skiprows=1, max_rows=3)
                    T = np.loadtxt(f, skiprows=4)
                    params.append((S, R, T))
        return params, txt_files

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
    print('--------- Create reenactment NMFC --------- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default='datasets/head2headDataset/dataset',
                        help='Path to the dataset directory.')
    parser.add_argument('--split_s', type=str,
                        default='train',
                        help='Split were source identity belongs.')
    parser.add_argument('--split_t', type=str,
                        default='train',
                        help='Split were target identity belongs.')
    parser.add_argument('--source_id', type=str,
                        default='Obama',
                        help='Id/name of the source person.')
    parser.add_argument('--target_id', type=str,
                        default='Trudeau',
                        help='Id/name of the target person.')
    parser.add_argument('--adapt_scale', action='store_true',
                        default=True,
                        help='Perform scale adaptation using statistics \
                              from target video.')
    parser.add_argument('--adapt_scale_and_translation', action='store_true',
                        default=True,
                        help='Perform scale and adaptation standardization \
                              using statistics from target video.')
    parser.add_argument('--standardize', action='store_true',
                        help='Perform adaptation using std from videos.')
    parser.add_argument('--gpu_id', type=int,
                        default='0',
                        help='Negative value to use CPU, or greater equal than \
                              zero for GPU id.')
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
    # Remove '_' from id names.
    args.source_id = args.source_id.replace('_', '')
    args.target_id = args.target_id.replace('_', '')
    # Print Arguments
    print_args(parser, args)
    # Initialize the NMFC renderer.
    renderer = NMFCRenderer(args)
    # Read the expression parameters from the source person.
    exp_params, paths = read_params('exp', os.path.join(args.dataset_path,
                                    args.split_s, 'exp_coeffs'), args.source_id)
    # Read the identity parameters from the target person.
    id_params, _ = read_params('id', os.path.join(args.dataset_path,
                               args.split_t, 'id_coeffs'), args.target_id)
    id_params = [id_params] * len(exp_params)
    # Read camera parameters from source
    s_cam_params, _ = read_params('cam', os.path.join(args.dataset_path,
                                  args.split_s, 'misc'), args.source_id)
    # Read camera parameters from target
    t_cam_params, _ = read_params('cam', os.path.join(args.dataset_path,
                                  args.split_t, 'misc'), args.target_id)
    # Compute the camera parameters.
    cam_params = compute_cam_params(s_cam_params, t_cam_params, args)
    source_nmfcs = renderer.computeNMFCs(cam_params, id_params, exp_params)
    source_images_paths = [os.path.splitext(path.replace('exp_coeffs',
                           'images'))[0] + '.png' for path in paths]
    save_results(source_nmfcs, source_images_paths, args)
    # Clean
    renderer.clear()

if __name__=='__main__':
    main()
