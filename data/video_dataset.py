import cv2
import os
import random
import torch
import numpy as np
import torchvision
from PIL import Image
from scipy.optimize import curve_fit
from data.base_dataset import BaseDataset, get_params, get_transform, get_video_parameters
from data.image_folder import make_video_dataset, assert_valid_pairs

class videoDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        do_reenactment = opt.do_reenactment if not opt.isTrain else False
        prefix = 'source_' if do_reenactment else ''
        source_name = opt.source_name if not opt.isTrain else None

        # Get dataset directories.
        self.dir_nmfc_video = os.path.join(opt.dataroot, self.opt.phase, prefix + 'nmfcs')
        self.nmfc_video_paths = make_video_dataset(self.dir_nmfc_video, opt.target_name, source_name)
        self.dir_rgb_video = os.path.join(opt.dataroot, self.opt.phase, prefix + 'images')
        self.rgb_video_paths = make_video_dataset(self.dir_rgb_video, opt.target_name, source_name)
        assert_valid_pairs(self.nmfc_video_paths, self.rgb_video_paths)
        if not opt.no_eye_gaze or (self.opt.use_mouth_D and self.opt.isTrain) or (self.opt.use_eyes_D and self.opt.isTrain):
            self.dir_landmark_video = os.path.join(opt.dataroot, self.opt.phase, prefix + 'landmarks70')
            self.landmark_video_paths = make_video_dataset(self.dir_landmark_video, opt.target_name, source_name)
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
        if not self.opt.no_eye_gaze or (self.opt.use_mouth_D and self.opt.isTrain) or (self.opt.use_eyes_D and self.opt.isTrain):
            landmark_video_paths = self.landmark_video_paths[seq_idx]

        # Get parameters and transforms.
        n_frames_total, start_idx = get_video_parameters(self.opt, self.n_frames_total, nmfc_len, self.frame_idx)
        first_nmfc_image = Image.open(nmfc_video_paths[0]).convert('RGB')
        params = get_params(self.opt, first_nmfc_image.size)
        transform_scale_nmfc_video = get_transform(self.opt, params, normalize=False,
            augment=not self.opt.no_augment_input and self.opt.isTrain) # do not normalize nmfc but augment.
        transform_scale_eye_gaze_video = get_transform(self.opt, params, normalize=False) # do not normalize eye_gaze.
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
                eye_video_i = self.create_eyes_image(landmark_video_path, first_nmfc_image.size,
                                                     transform_scale_eye_gaze_video,
                                                     add_noise=self.opt.isTrain)
                eye_video = eye_video_i if i == 0 else torch.cat([eye_video, eye_video_i], dim=0)
            if self.opt.use_mouth_D and self.opt.isTrain:
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

    def create_eyes_image(self, A_path, size, transform_scale, add_noise):
        def func(x, a, b, c):
            return a * x**2 + b * x + c

        def linear(x, a, b):
            return a * x + b

        def setColor(im, yy, xx, color):
            if len(im.shape) == 3:
                im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]
                #if (im[yy, xx] == 0).all():
                #    im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]
                #else:
                #    im[yy, xx, 0] = ((im[yy, xx, 0].astype(float) + color[0]) / 2).astype(np.uint8)
                #    im[yy, xx, 1] = ((im[yy, xx, 1].astype(float) + color[1]) / 2).astype(np.uint8)
                #    im[yy, xx, 2] = ((im[yy, xx, 2].astype(float) + color[2]) / 2).astype(np.uint8)
            else:
                im[yy, xx] = color[0]

        def drawCircle(im, x, y, rad, color=(255,0,0)):
            if x is not None and x.size:
                h, w = im.shape[0], im.shape[1]
                # edge
                for i in range(-rad, rad):
                    for j in range(-rad, rad):
                        yy = np.maximum(0, np.minimum(h-1, y+i))
                        xx = np.maximum(0, np.minimum(w-1, x+j))
                        if np.linalg.norm(np.array([i, j])) < rad:
                            setColor(im, yy, xx, color)

        def drawEdge(im, x, y, bw=1, color=(255,255,255)):
            if x is not None and x.size:
                h, w = im.shape[0], im.shape[1]
                # edge
                for i in range(-bw, bw):
                    for j in range(-bw, bw):
                        yy = np.maximum(0, np.minimum(h-1, y+i))
                        xx = np.maximum(0, np.minimum(w-1, x+j))
                        setColor(im, yy, xx, color)

        def interpPoints(x, y):
            if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
                curve_y, curve_x = interpPoints(y, x)
                if curve_y is None:
                    return None, None
            else:
                if len(x) < 3:
                    popt, _ = curve_fit(linear, x, y)
                else:
                    popt, _ = curve_fit(func, x, y)
                    if abs(popt[0]) > 1:
                        return None, None
                if x[0] > x[-1]:
                    x = list(reversed(x))
                    y = list(reversed(y))
                curve_x = np.linspace(x[0], x[-1], (x[-1]-x[0]))
                if len(x) < 3:
                    curve_y = linear(curve_x, *popt)
                else:
                    curve_y = func(curve_x, *popt)
            return curve_x.astype(int), curve_y.astype(int)

        w, h = size
        eyes_image = np.zeros((h, w, 3), np.int32)
        keypoints = np.loadtxt(A_path, delimiter=' ')
        if keypoints.shape[0] == 70:
            # if all 70 landmarks are available get only 14 for the eyes
            pts0 = keypoints[36:48, :].astype(np.int32) # eyes landmarks from 70 landmarks
            pts1 = keypoints[68:70, :].astype(np.int32) # eyes landmarks from 70 landmarks
            pts = np.concatenate([pts0, pts1], axis=0)
        else:
            pts = keypoints.astype(np.int32)
        left_eye_pts = np.concatenate([pts[0:6, :], pts[12:13, :]], axis=0)
        right_eye_pts = np.concatenate([pts[6:12, :], pts[13:14, :]], axis=0)

        if add_noise:
            scale_noise = 2 * np.random.randn(1)
            scale = 1 + scale_noise[0] / 100
            left_eye_mean = np.mean(left_eye_pts, axis=0, keepdims=True)
            right_eye_mean = np.mean(right_eye_pts, axis=0, keepdims=True)
            left_eye_pts = (left_eye_pts - left_eye_mean) * scale + left_eye_mean
            right_eye_pts = (right_eye_pts - right_eye_mean) * scale + right_eye_mean
            # add noise to eyes distance (x dimension)
            d_noise = 2 * np.random.randn(2)
            left_eye_pts += d_noise[0]
            right_eye_pts -= d_noise[1]

        pts = np.concatenate([left_eye_pts, right_eye_pts], axis=0).astype(np.int32)

        # Draw
        face_list = [ [[0,1,2,3], [3,4,5,0]], # left eye
                      [[7,8,9,10], [10,11,12,7]], # right eye
                     ]
        for edge_list in face_list:
                for edge in edge_list:
                    for i in range(0, max(1, len(edge)-1)):
                        sub_edge = edge[i:i+2]
                        x, y = pts[sub_edge, 0], pts[sub_edge, 1]
                        curve_x, curve_y = interpPoints(x, y)
                        drawEdge(eyes_image, curve_x, curve_y)
        radius_left = (np.linalg.norm(pts[1]-pts[4]) + np.linalg.norm(pts[2]-pts[5])) / 4
        radius_right = (np.linalg.norm(pts[8]-pts[11]) + np.linalg.norm(pts[9]-pts[12])) / 4
        drawCircle(eyes_image, pts[6, 0], pts[6, 1], int(radius_left))
        drawCircle(eyes_image, pts[13, 0], pts[13, 1], int(radius_right))
        eyes_image = transform_scale(Image.fromarray(np.uint8(eyes_image)))
        return eyes_image

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
