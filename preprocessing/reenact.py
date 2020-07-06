import cv2
import os
import numpy as np
import torch
import argparse
import sys
import scipy.io as io
from shutil import copyfile
import itertools
from preprocessing.reconstruction import NMFCRenderer

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
        # Save them as 70 landmarks, even they are actually only eye landmarks.
        save_landmarks70_dir = os.path.join(args.dataset_path, 'test',
                            'source_landmarks70', args.target_id + '_' + args.source_id)
        mkdirs([save_landmarks70_dir])
    for i, source_images_path in enumerate(source_images_paths):
        frame_name = os.path.basename(source_images_path)
        copyfile(source_images_path, os.path.join(save_images_dir, frame_name))
        cv2.imwrite(os.path.join(save_nmfcs_dir, frame_name), nmfcs[i])
        if eye_landmarks is not None:
            np.savetxt(os.path.join(save_landmarks70_dir, os.path.splitext(frame_name)[0] + '.txt'), eye_landmarks[i])

def smoothen_signal(S, window_size=15):
    left_p = window_size // 2
    right_p =  window_size // 2 if window_size % 2 == 1 else window_size // 2 - 1
    window = np.ones(int(window_size))/float(window_size) # kernel-filter
    S = np.array(S)
    # Padding
    left_padding = np.stack([S[0]] * left_p, axis=0)
    right_padding = np.stack([S[-1]] * right_p, axis=0)
    S_padded = np.concatenate([left_padding, S, right_padding])
    if len(S_padded.shape) == 1:
        S = np.convolve(S_padded, window, 'valid')
    else:
        for coord in range(S_padded.shape[1]):
            S[:, coord] = np.convolve(S_padded[:, coord], window, 'valid')
    return S

def compute_cam_params(s_cam_params, t_cam_params, args):
    cam_params = s_cam_params
    if not args.no_scale_or_translation_adaptation:
        mean_S_target = np.mean([params[0] for params in t_cam_params])
        mean_S_source = np.mean([params[0] for params in s_cam_params])
        S = [params[0] * (mean_S_target / mean_S_source)
             for params in s_cam_params]
        # Smoothen scale
        S = smoothen_signal(S)
        # Normalised Translation for source and target.
        nT_target = [params[2] / params[0] for params in t_cam_params]
        nT_source = [params[2] / params[0] for params in s_cam_params]
        cam_params = [(s, params[1], s * t) \
                      for s, params, t in zip(S, s_cam_params, nT_source)]
        if not args.no_translation_adaptation:
            mean_nT_target = np.mean(nT_target, axis=0)
            mean_nT_source = np.mean(nT_source, axis=0)
            std_nT_target = np.std(nT_target, axis=0)
            # Allow camera translation two standard deviation away from the one on target video.
            upper_limit = mean_nT_target + std_nT_target * 2
            lower_limit = mean_nT_target - std_nT_target * 2
            nT = [np.maximum(np.minimum(t - mean_nT_source + mean_nT_target,
                             upper_limit), lower_limit) for t in nT_source]
            # Smoothen translation
            nT = smoothen_signal(nT)
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
            left = np.concatenate([np.loadtxt(f)[36:42], np.loadtxt(f)[68:69]], axis=0)
            right = np.concatenate([np.loadtxt(f)[42:48], np.loadtxt(f)[69:70]], axis=0)
            eye_landmarks_left.append(left)  # Left eye
            eye_landmarks_right.append(right) # Right eye
    return [eye_landmarks_left, eye_landmarks_right]

def search_eye_centres(nmfcs, prev_arg_mins=None):
    points = [np.array([192, 180, 81]), # Left eye NMFC code
              np.array([192, 180, 171])] # Right eye NMFC code
    ret = []
    arg_mins = []
    if prev_arg_mins is None:
        prev_arg_mins = [None, None]
    for point, prev_arg_min in zip(points, prev_arg_mins):
        centres = []
        for n, nmfc in enumerate(nmfcs):
            min_dst = 99999999
            if prev_arg_min is None:
                lim_i_l, lim_i_h = 0, nmfc.shape[0]-1
                lim_j_l, lim_j_h = 0, nmfc.shape[1]-1
            else:
                lim_i_l, lim_i_h = prev_arg_min[0]-20, prev_arg_min[0]+20
                lim_j_l, lim_j_h = prev_arg_min[1]-20, prev_arg_min[1]+20
            # Check bounds
            lim_i_l = min(max(lim_i_l, 0), nmfc.shape[0]-1)
            lim_i_h = min(max(lim_i_h, 0), nmfc.shape[0]-1)
            lim_j_l = min(max(lim_j_l, 0), nmfc.shape[1]-1)
            lim_j_h = min(max(lim_j_h, 0), nmfc.shape[1]-1)
            for i in range(lim_i_l, lim_i_h):
                for j in range(lim_j_l, lim_j_h):
                    dst = sum(abs(nmfc[i,j,:] - point))
                    if dst < min_dst:
                        min_dst = dst
                        arg_min = np.array([i, j])
            centres.append(np.flip(arg_min)) # flip, since landmarks are width, heigth
            prev_arg_min = arg_min
        arg_mins.append(arg_min)
        ret.append(centres)
    return ret, arg_mins

def smoothen_eye_landmarks(eye_landmarks, window_size=1):
    window_size = max(min(window_size, len(eye_landmarks)), 1)
    left_p = window_size // 2
    right_p =  window_size // 2 if window_size % 2 == 1 else window_size // 2 - 1
    window = np.ones(int(window_size))/float(window_size) # kernel-filter
    eye_landmarks = np.array(eye_landmarks)
    # Padding
    left_padding = np.stack([eye_landmarks[0]] * left_p, axis=0) if left_p > 0 else None
    right_padding = np.stack([eye_landmarks[-1]] * right_p, axis=0) if right_p > 0 else None
    eye_landmarks_padded = eye_landmarks
    if left_padding is not None:
        eye_landmarks_padded = np.concatenate([left_padding, eye_landmarks_padded])
    if right_padding is not None:
        eye_landmarks_padded = np.concatenate([eye_landmarks_padded, right_padding])
    for land in range(eye_landmarks.shape[1]):
        for coord in range(eye_landmarks.shape[2]):
            eye_landmarks[:, land, coord] = np.convolve(eye_landmarks_padded[:, land, coord], window, 'valid')
    return eye_landmarks

def compute_eye_landmarks_ratio(eye_landmarks_source, eye_landmarks_target):
    dsts = []
    for eye_landmarks in [eye_landmarks_source, eye_landmarks_target]:
        each_eye_dsts = []
        for each_eye_landmarks in eye_landmarks:
            dst = 0
            for each_eye_landmark in each_eye_landmarks:
                eye_width = np.linalg.norm(each_eye_landmark[0,:] - each_eye_landmark[3,:])
                dst += (abs(each_eye_landmark[1, 1] - each_eye_landmark[4, 1]) + \
                       abs(each_eye_landmark[2, 1] - each_eye_landmark[5, 1])) #/ eye_width
            each_eye_dsts.append(dst / len(each_eye_landmarks))
        dsts.append(each_eye_dsts)
    left_eye_ratio = dsts[1][0] / dsts[0][0]
    right_eye_ratio = dsts[1][1] / dsts[0][1]
    return [left_eye_ratio, right_eye_ratio]

def adapt_eye_landmarks(eye_landmarks, eye_centres, eye_ratios, s_cam_params, cam_params):
    new_eye_landmarks = []
    ratios = [cam_param[0] / s_cam_param[0]
                for s_cam_param, cam_param in zip(s_cam_params, cam_params)]
    for each_eye_landmarks, each_eye_centres, each_eye_ratios in zip(eye_landmarks, eye_centres, eye_ratios):
        new_each_eye_landmarks = []
        for each_eye_landmark, each_eye_centre, ratio in zip(each_eye_landmarks, each_eye_centres, ratios):
            mean = np.mean(each_eye_landmark, axis=0, keepdims=True)
            new_each_eye_landmark = (each_eye_landmark - mean) * ratio
            new_each_eye_landmark[1:3, 1] *= each_eye_ratios / ratio
            new_each_eye_landmark[4:6, 1] *= each_eye_ratios / ratio
            new_each_eye_landmark += np.expand_dims(each_eye_centre, axis=0)
            new_each_eye_landmarks.append(new_each_eye_landmark)
        new_eye_landmarks.append(new_each_eye_landmarks)
    ret_eye_landmarks = []
    for left_eye_landmarks, right_eye_landmarks in zip(new_eye_landmarks[0], new_eye_landmarks[1]):
        ret_eye_landmarks.append(np.concatenate([left_eye_landmarks[0:6], right_eye_landmarks[0:6],
                                                 left_eye_landmarks[6:7], right_eye_landmarks[6:7]], axis=0).astype(np.int32))
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
    parser.add_argument('--keep_target_pose', action='store_true',
                        help='Use the poses from target video.')
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
    # Check if conditional input files already exist. (We check only NMFCs)
    save_nmfcs_dir = os.path.join(args.dataset_path, 'test',
                    'source_nmfcs', args.target_id + '_' + args.source_id)
    if os.path.isdir(save_nmfcs_dir):
        print('Conditional input files already exist!')
        exit(0)
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
    if args.keep_target_pose:
        s_len = len(s_cam_params)
        t_len = len(t_cam_params)
        if s_len <= t_len:
            cam_params = t_cam_params[0:s_len]
        else:
            n_repeat = (s_len - 1) // t_len + 1
            cam_params = [t_cam_params] * n_repeat
            cam_params = list(itertools.chain.from_iterable(cam_params))
            cam_params = cam_params[0:s_len]
    else:
        # Compute the camera parameters.
        cam_params = compute_cam_params(s_cam_params, t_cam_params, args)
    # Create NMFC images
    nmfcs = renderer.computeNMFCs(cam_params, id_params, exp_params)
    # Create Eye landmarks
    eye_landmarks = None
    if not args.no_eye_gaze:
        eye_landmarks_target = read_eye_landmarks(os.path.join(args.dataset_path,
                                args.split_t, 'landmarks70'), args.target_id)
        if args.keep_target_pose:
            if s_len <= t_len:
                eye_landmarks_source = eye_landmarks_target[0:s_len]
            else:
                n_repeat = (s_len - 1) // t_len + 1
                eye_landmarks_source = [eye_landmarks_target] * n_repeat
                eye_landmarks_source = list(itertools.chain.from_iterable(eye_landmarks_source))
                eye_landmarks_source = eye_landmarks_source[0:s_len]
        else:
            eye_landmarks_source = read_eye_landmarks(os.path.join(args.dataset_path,
                                    args.split_s, 'landmarks70'), args.source_id)
        eye_centres, _ = search_eye_centres(nmfcs)
        eye_ratios = compute_eye_landmarks_ratio(eye_landmarks_source,
                                                 eye_landmarks_target)
        eye_landmarks = adapt_eye_landmarks(eye_landmarks_source, eye_centres, eye_ratios,
                                            s_cam_params, cam_params)
        if not args.keep_target_pose:
            eye_landmarks = smoothen_eye_landmarks(eye_landmarks)
    source_images_paths = [os.path.splitext(path.replace('exp_coeffs',
                           'images'))[0] + '.png' for path in paths]
    save_results(nmfcs, eye_landmarks, source_images_paths, args)
    # Clean
    renderer.clear()

if __name__=='__main__':
    main()
