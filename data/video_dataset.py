import cv2
import os
import random
import torch
import numpy as np
import torchvision
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform, get_video_parameters
from data.image_folder import make_video_dataset, assert_valid_pairs

class videoDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.do_reenactment = opt.do_reenactment if not opt.isTrain else False
        prefix = 'source_' if self.do_reenactment else ''

        # Get dataset directories.
        self.dir_nmfc_video = os.path.join(opt.dataroot, self.opt.phase, prefix + 'nmfcs')
        self.nmfc_video_paths = make_video_dataset(self.dir_nmfc_video, opt.target_name)
        self.dir_rgb_video = os.path.join(opt.dataroot, self.opt.phase, prefix + 'images')
        self.rgb_video_paths = make_video_dataset(self.dir_rgb_video, opt.target_name)
        assert_valid_pairs(self.nmfc_video_paths, self.rgb_video_paths)
        if not opt.no_eye_gaze:
            self.dir_eye_video = os.path.join(opt.dataroot, self.opt.phase, prefix + 'eyes')
            self.eye_video_paths = make_video_dataset(self.dir_eye_video, opt.target_name)
            assert_valid_pairs(self.eye_video_paths, self.rgb_video_paths)
        if self.opt.finetune_mouth and self.opt.isTrain:
            self.dir_landmark_video = os.path.join(opt.dataroot, self.opt.phase, prefix + 'landmarks')
            self.landmark_video_paths = make_video_dataset(self.dir_landmark_video, opt.target_name)
            assert_valid_pairs(self.landmark_video_paths, self.rgb_video_paths)

        self.n_of_seqs = len(self.nmfc_video_paths)
        self.seq_len_max = max([len(A) for A in self.nmfc_video_paths])
        self.init_frame_index(self.nmfc_video_paths)

    def __getitem__(self, index):
        # Get sequence paths.
        seq_idx = self.update_frame_index(self.nmfc_video_paths, index)
        nmfc_video_paths = self.nmfc_video_paths[seq_idx]
        nmfc_len = len(nmfc_video_paths)
        rgb_video_paths = self.rgb_video_paths[seq_idx]
        if not self.opt.no_eye_gaze:
            eye_video_paths = self.eye_video_paths[seq_idx]
        if self.opt.finetune_mouth and self.opt.isTrain:
            landmark_video_paths = self.landmark_video_paths[seq_idx]

        # Get parameters and transforms.
        n_frames_total, start_idx = get_video_parameters(self.opt, self.n_frames_total, nmfc_len, self.frame_idx)
        first_nmfc_image = Image.open(nmfc_video_paths[0]).convert('RGB')
        params = get_params(self.opt, first_nmfc_image.size)
        transform_scale_nmfc_video = get_transform(self.opt, params, normalize=False) # do not normalize nmfc values
        transform_scale_rgb_video = get_transform(self.opt, params)
        change_seq = False if self.opt.isTrain else self.change_seq

        # Read data.
        A_paths = []
        rgb_video = nmfc_video = eye_video = mouth_centers = 0
        for i in range(n_frames_total):
            # NMFC
            nmfc_video_path = nmfc_video_paths[start_idx + i]
            nmfc_video_i = self.get_image(nmfc_video_path, transform_scale_nmfc_video)
            nmfc_video = nmfc_video_i if i == 0 else torch.cat([nmfc_video, nmfc_video_i], dim=0)
            # RGB
            rgb_video_path = rgb_video_paths[start_idx + i]
            rgb_video_i = self.get_image(rgb_video_path, transform_scale_rgb_video)
            rgb_video = rgb_video_i if i == 0 else torch.cat([rgb_video, rgb_video_i], dim=0)
            A_paths.append(nmfc_video_path)
            if not self.opt.no_eye_gaze:
                eye_video_path = eye_video_paths[start_idx + i]
                eye_video_i = self.get_image(eye_video_path, transform_scale_nmfc_video)
                eye_video = eye_video_i if i == 0 else torch.cat([eye_video, eye_video_i], dim=0)
            if self.opt.finetune_mouth and self.opt.isTrain:
                landmark_video_path = landmark_video_paths[start_idx + i]
                mouth_centers_i = self.read_mouth_keypoints(landmark_video_path, ratio=(1,1))
                mouth_centers = mouth_centers_i if i == 0 else torch.cat([mouth_centers, mouth_centers_i], dim=0)

        return_list = {'nmfc_video': nmfc_video, 'rgb_video':rgb_video,
                       'eye_video':eye_video, 'mouth_centers':mouth_centers,
                       'change_seq':change_seq, 'A_paths':A_paths}
        return return_list

    def read_mouth_keypoints(self, A_path, ratio):
        keypoints = np.loadtxt(A_path, delimiter=' ')
        if keypoints.shape[0] == 5:
            pts = keypoints[3:, :].astype(np.int32) # mouth landmarks Retinaface
            mouth_center = np.mean(pts, axis=0)
        else:
            pts = keypoints[48:, :].astype(np.int32) # mouth landmarks from 68 landmarks
            mouth_center = np.median(pts, axis=0)
        mouth_center[0] = ratio[0] * mouth_center[0]
        mouth_center[1] = ratio[1] * mouth_center[1]
        mouth_center = mouth_center.astype(np.int32)
        return torch.tensor(np.expand_dims(mouth_center, axis=0))

    def get_image(self, A_path, transform_scaleA, convert_rgb=True):
        A_img = Image.open(A_path)
        if convert_rgb:
            A_img = A_img.convert('RGB')
        A_scaled = transform_scaleA(A_img)
        return A_scaled

    def __len__(self):
        if self.opt.isTrain:
            return len(self.nmfc_video_paths)
        else:
            return sum(self.n_frames_in_sequence)

    def name(self):
        return 'nmfc'
