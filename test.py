import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

opt = TestOptions().parse(save=False)
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True

visualizer = Visualizer(opt)

modelG = create_model(opt)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

print('Generating %d frames' % dataset_size)

save_dir = os.path.join(opt.results_dir, opt.name, opt.which_epoch + '_epoch', opt.phase)

total_distance, total_pixels = 0, 0
mtotal_distance, mtotal_pixels = 0, 0
mouth_total_distance, mouth_total_pixels = 0, 0

for i, data in enumerate(dataset):
    if opt.time_fwd_pass:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    if data['change_seq']:
        modelG.fake_B_prev = None

    _, _, height, width = data['nmfc_video'].size()
    nmfc_video = Variable(data['nmfc_video']).view(1, -1, 3, height, width)
    input_A = nmfc_video
    rgb_video = Variable(data['rgb_video']).view(1, -1, 3, height, width)
    if not opt.no_eye_gaze:
        eye_gaze_video = Variable(data['eye_video']).view(1, -1, 3, height, width)
        input_A = torch.cat([nmfc_video, eye_gaze_video], dim=2)
    if not opt.do_reenactment:
        mouth_centers = Variable(data['mouth_centers']).view(1, -1, 2)
    img_path = data['A_paths']

    print('Processing NMFC image %s' % img_path[-1])

    generated = modelG.inference(input_A, rgb_video)

    if opt.time_fwd_pass:
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        print('Forward pass time: %.6f' % start.elapsed_time(end))

    fake_frame = util.tensor2im(generated[0].data[0])
    rgb_frame = util.tensor2im(rgb_video[0, -1])
    nmfc_frame = util.tensor2im(nmfc_video[0, -1], normalize=False)
    if not opt.no_eye_gaze:
        eye_gaze_frame = util.tensor2im(eye_gaze_video[0, -1], normalize=False)
    if not opt.do_reenactment:
        fake_frame = util.tensor2im(generated[0].data[0])
        rgb_frame_mouth = util.tensor2im(rgb_video[0, -1])
        mc = util.fit_ROI_in_frame(mouth_centers[0, -1], opt)
        fake_frame_mouth = util.tensor2im(util.crop_ROI(generated[0].data[0], mc, opt.ROI_size))
        rgb_frame_mouth = util.tensor2im(util.crop_ROI(rgb_video[0, -1], mc, opt.ROI_size))

    visual_list = [('real', rgb_frame),
                   ('fake', fake_frame),
                   ('nmfc', nmfc_frame)]
    if not opt.no_eye_gaze:
        visual_list += [('eye_gaze', eye_gaze_frame)]

    # If in self reenactment mode, compute pixel error and heatmap.
    if not opt.do_reenactment:
        total_distance, total_pixels, heatmap = util.get_pixel_distance(
                rgb_frame, fake_frame, total_distance, total_pixels)
        mtotal_distance, mtotal_pixels, mheatmap = util.get_pixel_distance(
            rgb_frame, fake_frame, mtotal_distance, mtotal_pixels, nmfc_frame)
        mouth_total_distance, mouth_total_pixels, mouth_heatmap = util.get_pixel_distance(
            rgb_frame_mouth, fake_frame_mouth, mouth_total_distance, mouth_total_pixels)
        visual_list += [('heatmap', heatmap),
                        ('masked_heatmap', mheatmap)]

    visuals = OrderedDict(visual_list)
    visualizer.save_images(save_dir, visuals, img_path[-1])

if not opt.do_reenactment:
    # Average Pixel Distance (APD-L2)
    print('Average pixel (L2) distance for sequence (APD-L2): %0.2f' % (total_distance/total_pixels))
    # Masked Average Pixel Distance (MAPD-L2)
    print('Masked average pixel (L2) distance for sequence (MAPD-L2): %0.2f' % (mtotal_distance/mtotal_pixels))
    # Average Pixel Distance in mouth region (mouth-APD-L2)
    print('Average pixel (L2) distance in mouth region for sequence (mouth-APD-L2): %0.2f' % (mouth_total_distance/mouth_total_pixels))
