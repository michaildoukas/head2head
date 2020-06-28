import os
import random
import torch
import numpy as np
import torchvision
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform, get_video_parameters
from data.image_folder import make_video_dataset, assert_valid_pairs
from data.landmarks_to_image import create_eyes_image

class videoDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        do_reenactment = opt.do_reenactment if not opt.isTrain else False
        prefix = 'source_' if do_reenactment else ''
        source_name = opt.source_name if not opt.isTrain else None

        # Get dataset directories.
        self.dir_nmfc_video = os.path.join(opt.dataroot, self.opt.phase, prefix + 'nmfcs')
        self.nmfc_video_paths = make_video_dataset(self.dir_nmfc_video, opt.target_name, source_name, opt.max_n_sequences)
        self.dir_rgb_video = os.path.join(opt.dataroot, self.opt.phase, prefix + 'images')
        self.rgb_video_paths = make_video_dataset(self.dir_rgb_video, opt.target_name, source_name, opt.max_n_sequences)
        assert_valid_pairs(self.nmfc_video_paths, self.rgb_video_paths)
        if not opt.no_eye_gaze or (not self.opt.no_mouth_D and self.opt.isTrain) or (self.opt.use_eyes_D and self.opt.isTrain):
            self.dir_landmark_video = os.path.join(opt.dataroot, self.opt.phase, prefix + 'landmarks70')
            self.landmark_video_paths = make_video_dataset(self.dir_landmark_video, opt.target_name, source_name, opt.max_n_sequences)
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
        if not self.opt.no_eye_gaze or (not self.opt.no_mouth_D and self.opt.isTrain) or (self.opt.use_eyes_D and self.opt.isTrain):
            landmark_video_paths = self.landmark_video_paths[seq_idx]

        # Get parameters and transforms.
        n_frames_total, start_idx = get_video_parameters(self.opt, self.n_frames_total, nmfc_len, self.frame_idx)
        first_nmfc_image = Image.open(nmfc_video_paths[0]).convert('RGB')
        params = get_params(self.opt, first_nmfc_image.size)
        transform_scale_nmfc_video = get_transform(self.opt, params, normalize=False,
            augment=not self.opt.no_augment_input and self.opt.isTrain) # do not normalize nmfc but augment.
        transform_scale_eye_gaze_video = transform_scale_nmfc_video #get_transform(self.opt, params, normalize=False) # do not normalize eye_gaze.
        transform_scale_rgb_video = get_transform(self.opt, params)
        change_seq = False if self.opt.isTrain else self.change_seq

        # Read data.
        A_paths = []
        rgb_video = nmfc_video = eye_video = mouth_centers = eyes_centers = 0
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
                landmark_video_path = landmark_video_paths[start_idx + i]
                eye_video_i = create_eyes_image(landmark_video_path, first_nmfc_image.size,
                                                transform_scale_eye_gaze_video,
                                                add_noise=self.opt.isTrain)
                eye_video = eye_video_i if i == 0 else torch.cat([eye_video, eye_video_i], dim=0)
            if not self.opt.no_mouth_D and self.opt.isTrain:
                landmark_video_path = landmark_video_paths[start_idx + i]
                mouth_centers_i = self.get_mouth_center(landmark_video_path)
                mouth_centers = mouth_centers_i if i == 0 else torch.cat([mouth_centers, mouth_centers_i], dim=0)
            if self.opt.use_eyes_D and self.opt.isTrain:
                landmark_video_path = landmark_video_paths[start_idx + i]
                eyes_centers_i = self.get_eyes_center(landmark_video_path)
                eyes_centers = eyes_centers_i if i == 0 else torch.cat([eyes_centers, eyes_centers_i], dim=0)

        return_list = {'nmfc_video': nmfc_video, 'rgb_video':rgb_video,
                       'eye_video':eye_video, 'mouth_centers':mouth_centers, 'eyes_centers':eyes_centers,
                       'change_seq':change_seq, 'A_paths':A_paths}
        return return_list

    def get_mouth_center(self, A_path):
        keypoints = np.loadtxt(A_path, delimiter=' ')
        pts = keypoints[48:, :].astype(np.int32) # mouth landmarks from 70 landmarks
        mouth_center = np.median(pts, axis=0)
        mouth_center = mouth_center.astype(np.int32)
        return torch.tensor(np.expand_dims(mouth_center, axis=0))

    def get_eyes_center(self, A_path):
        keypoints = np.loadtxt(A_path, delimiter=' ')
        if keypoints.shape[0] == 70:
            # if all 70 landmarks are available get only 14 for the eyes
            pts0 = keypoints[36:48, :].astype(np.int32) # eyes landmarks from 70 landmarks
            pts1 = keypoints[68:70, :].astype(np.int32) # eyes landmarks from 70 landmarks
            pts = np.concatenate([pts0, pts1], axis=0)
        else:
            pts = keypoints.astype(np.int32)
        eyes_center = np.median(pts, axis=0)
        eyes_center = eyes_center.astype(np.int32)
        return torch.tensor(np.expand_dims(eyes_center, axis=0))

    def get_image(self, A_path, transform_scale, convert_rgb=True):
        A_img = Image.open(A_path)
        if convert_rgb:
            A_img = A_img.convert('RGB')
        A_scaled = transform_scale(A_img)
        return A_scaled

    def __len__(self):
        if self.opt.isTrain:
            return len(self.nmfc_video_paths)
        else:
            return sum(self.n_frames_in_sequence)

    def name(self):
        return 'nmfc'
