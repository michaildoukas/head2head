import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--max_n_sequences', type=int, default=None, help='Maximum number of sub-sequences to use.')
        self.parser.add_argument('--no_augment_input', action='store_true', help='if true, do not perform input data augmentation.')
        self.parser.add_argument('--ROI_size', type=int, default=72, help='spatial dimension size of ROI (mouth or eyes).')
        self.parser.add_argument('--no_mouth_D', action='store_true', help='if true, do not use mouth discriminator')
        self.parser.add_argument('--use_eyes_D', action='store_true', help='if true, Use eyes discriminator')
        self.parser.add_argument('--no_eye_gaze', action='store_true', help='if true, the model does not condition synthesis on eye gaze images')
        self.parser.add_argument('--use_faceflow', action='store_true', default=True, help='if true, Use fine-tuned flow on faces.')
        self.parser.add_argument('--n_frames_G', type=int, default=3, help='number of input frames to feed into generator, i.e., n_frames_G-1 is the number of frames we look into past')
        self.parser.add_argument('--no_first_img', action='store_true', default=True, help='if specified, generator synthesizes the first image')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='if specified, load the pretrained model')
        self.parser.add_argument('--resize', action='store_true', default=True, help='Resize the input images to loadSize')

        self.parser.add_argument('--dataroot', type=str, default='datasets/head2headDataset/dataset')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--input_nc', type=int, default=6, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--dataset_mode', type=str, default='video', help='')
        self.parser.add_argument('--target_name', type=str, default=None, help='Name of target person. If None, train on all targets.')

        # network arch
        self.parser.add_argument('--no_prev_output', action='store_true', help='if true, do not use the previously generated frames in G input.')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--n_blocks', type=int, default=9, help='number of resnet blocks in generator')
        self.parser.add_argument('--n_downsample_G', type=int, default=3, help='number of downsampling layers in netG')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='head2head', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/head2headDataset', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, perform shuffling in path creation, otherwise in the dataloader. Set in case of frequent out of memory exceptions.')
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
