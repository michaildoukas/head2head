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

def save_results(nmfcs, eye_landmarks, source_images_paths, args):
    assert len(nmfcs) == len(source_images_paths), \
            'Rendered NMFC and original source sequence have different lengths.'
    if eye_landmarks is not None:
        assert len(eye_landmarks) == len(source_images_paths), \
                'Adapted eye landmark sequence and original source sequence have different lengths.'
    save_nmfcs_dir = os.path.join(args.dataset_path, 'test',
                        'source_nmfcs', args.target_id + '_' + args.source_id)
    save_images_dir = os.path.join(args.dataset_path, 'test',
                        'source_images', args.target_id + '_' + args.source_id)
    mkdirs([save_nmfcs_dir, save_images_dir])
    if eye_landmarks is not None:
        # Save them as 68 landmarks, even they are actually only eye landmarks.
        save_landmarks68_dir = os.path.join(args.dataset_path, 'test',
                            'source_landmarks68', args.target_id + '_' + args.source_id)
        mkdirs([save_landmarks68_dir])
    for i, source_images_path in enumerate(source_images_paths):
        frame_name = os.path.basename(source_images_path)
        copyfile(source_images_path, os.path.join(save_images_dir, frame_name))
        cv2.imwrite(os.path.join(save_nmfcs_dir, frame_name), nmfcs[i])
        if eye_landmarks is not None:
            np.savetxt(os.path.join(save_landmarks68_dir, os.path.splitext(frame_name)[0] + '.txt'), eye_landmarks[i])

def compute_cam_params(s_cam_params, t_cam_params, args):
    cam_params = s_cam_params
    if not args.no_scale_or_translation_adaptation:
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
        if not args.no_translation_adaptation:
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

def read_eye_landmarks(path, speaker_id):
    txt_files = []
    eye_landmarks_left = []
    eye_landmarks_right = []
    parts = os.listdir(path)
    base_part = os.path.join(path, speaker_id)
    for part in sorted(parts):
        dir = os.path.join(path, part)
        if base_part in dir:
            txt_files.extend([os.path.join(dir, txt) \
                             for txt in sorted(os.listdir(dir))])
    for f in txt_files:
        if os.path.exists(f):
            eye_landmarks_left.append(np.loadtxt(f)[36:42])  # Left eye
            eye_landmarks_right.append(np.loadtxt(f)[42:48]) # Right eye
    return [eye_landmarks_left, eye_landmarks_right]

def search_eye_centres(nmfcs):
    points = [np.array([192, 180, 81]), # Left eye NMFC code
              np.array([192, 180, 171])] # Right eye NMFC code
    ret = []
    for point in points:
        centres = []
        for n, nmfc in enumerate(nmfcs):
            min_dst = 99999999
            if n == 0:
                lim_i_l, lim_i_h = 0, nmfc.shape[0]-1
                lim_j_l, lim_j_h = 0, nmfc.shape[1]-1
            else:
                lim_i_l, lim_i_h = prev_arg_min[0]-20, prev_arg_min[0]+20
                lim_j_l, lim_j_h = prev_arg_min[1]-20, prev_arg_min[1]+20
            for i in range(lim_i_l, lim_i_h):
                for j in range(lim_j_l, lim_j_h):
                    dst = sum(abs(nmfc[i,j,:] - point))
                    if dst < min_dst:
                        min_dst = dst
                        arg_min = np.array([i, j])
            #print(min_dst)
            prev_arg_min = arg_min
            centres.append(np.flip(arg_min)) # flip, since landmarks are width, heigth
        ret.append(centres)
    return ret

def adapt_eye_landmarks(eye_landmarks, eye_centres, s_cam_params, cam_params):
    new_eye_landmarks = []
    ratios = [cam_param[0] / s_cam_param[0]
                for s_cam_param, cam_param in zip(s_cam_params, cam_params)]
    for each_eye_landmarks, each_eye_centres in zip(eye_landmarks, eye_centres):
        new_each_eye_landmarks = []
        for each_eye_landmark, each_eye_centre, ratio in zip(each_eye_landmarks, each_eye_centres, ratios):
            mean = np.mean(each_eye_landmark, axis=0, keepdims=True)
            new_each_eye_landmark = (each_eye_landmark - mean) * ratio + np.expand_dims(each_eye_centre, axis=0)
            new_each_eye_landmarks.append(new_each_eye_landmark)
        new_eye_landmarks.append(new_each_eye_landmarks)
    ret_eye_landmarks = []
    for left_eye_landmarks, right_eye_landmarks in zip(new_eye_landmarks[0], new_eye_landmarks[1]):
        ret_eye_landmarks.append(np.concatenate([left_eye_landmarks, right_eye_landmarks], axis=0).astype(np.int32))
    return ret_eye_landmarks

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
    parser.add_argument('--dataset_name', type=str,
                        default='head2headDataset',
                        help='Path to the dataset directory.')
    parser.add_argument('--split_s', type=str,
                        default='test',
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
    parser.add_argument('--no_scale_or_translation_adaptation', action='store_true',
                        help='Do not perform scale or translation adaptation \
                              using statistics from target video.')
    parser.add_argument('--no_translation_adaptation', action='store_true',
                        help='Do not perform translation adaptation \
                              using statistics from target video.')
    parser.add_argument('--standardize', action='store_true',
                        help='Perform adaptation using also std from videos.')
    parser.add_argument('--no_eye_gaze', action='store_true',
                        help='.')
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

    args.dataset_path = os.path.join('datasets', args.dataset_name, 'dataset')

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
    # Create NMFC images
    nmfcs = renderer.computeNMFCs(cam_params, id_params, exp_params)
    # Create Eye landmarks
    eye_landmarks = None
    if not args.no_eye_gaze:
        eye_landmarks = read_eye_landmarks(os.path.join(args.dataset_path,
                                args.split_s, 'landmarks68'), args.source_id)
        eye_centres = search_eye_centres(nmfcs)
        eye_landmarks = adapt_eye_landmarks(eye_landmarks, eye_centres,
                                            s_cam_params, cam_params)
    source_images_paths = [os.path.splitext(path.replace('exp_coeffs',
                           'images'))[0] + '.png' for path in paths]
    save_results(nmfcs, eye_landmarks, source_images_paths, args)
    # Clean
    renderer.clear()

if __name__=='__main__':
    main()
