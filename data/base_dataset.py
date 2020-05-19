import torch.utils.data as data
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

    def init_frame_index(self, A_paths):
        self.seq_idx = 0
        self.frame_idx = -1 if not self.opt.isTrain else 0
        self.n_frames_total = self.opt.n_frames_total if self.opt.isTrain else 1
        self.n_sequences = len(A_paths)
        self.max_seq_len = max([len(A) for A in A_paths])
        self.n_frames_in_sequence = []
        for path in A_paths:
            self.n_frames_in_sequence.append(len(path) - self.opt.n_frames_G + 1)

    def update_frame_index(self, A_paths, index):
        if self.opt.isTrain:
            seq_idx = index % self.n_sequences
            return seq_idx
        else:
            self.change_seq = self.frame_idx >= self.n_frames_in_sequence[self.seq_idx] - 1
            if self.change_seq:
                self.seq_idx += 1
                self.frame_idx = 0
            else:
                self.frame_idx += 1
            return self.seq_idx

    def update_sequence_length(self, ratio):
        max_seq_len = self.max_seq_len - self.opt.n_frames_G + 1
        if self.n_frames_total < max_seq_len:
            self.n_frames_total = min(max_seq_len, self.opt.n_frames_total * (2**ratio))
            print('Updated sequence length to %d' % self.n_frames_total)

def get_params(opt, size):
    w, h = size
    if opt.resize:
        new_h = new_w = opt.loadSize
        new_w = int(round(new_w / 4)) * 4
        new_h = int(round(new_h / 4)) * 4
        new_w, new_h = __make_power_2(new_w), __make_power_2(new_h)
    else:
        new_h = h
        new_w = w
    return {'new_size': (new_w, new_h), 'ratio':(new_h / h, new_w / w)}

def get_transform(opt, params, method=Image.BICUBIC, normalize=True, augment=False, toTensor=True):
    transform_list = []
    ### resize input image
    if opt.resize:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))
    else:
        transform_list.append(transforms.Lambda(lambda img: __scale(img, params['new_size'], method)))

    if augment:
        transform_list += [transforms.RandomAffine(degrees=(0, 0),
                                                   translate=(0.01, 0.01),
                                                   scale=(0.99, 1.01))]
    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale(img, size, method=Image.BICUBIC):
    w, h = size
    return img.resize((w, h), method)

def __make_power_2(n, base=32.0):
    return int(round(n / base) * base)

def get_video_parameters(opt, n_frames_total, cur_seq_len, index):
    if opt.isTrain:
        n_frames_total = min(n_frames_total, cur_seq_len - opt.n_frames_G + 1)
        n_frames_total += opt.n_frames_G - 1
        offset_max = max(1, cur_seq_len - n_frames_total + 1)
        start_idx = np.random.randint(offset_max)
    else:
        n_frames_total = opt.n_frames_G
        start_idx = index
    return n_frames_total, start_idx
