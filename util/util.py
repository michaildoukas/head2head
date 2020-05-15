from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
from PIL import Image
import cv2
from scipy.spatial import distance

def reshape(tensors):
    if isinstance(tensors, list):
        return [reshape(tensor) for tensor in tensors]
    if tensors is None:
        return None
    _, _, ch, h, w = tensors.size()
    return tensors.contiguous().view(-1, ch, h, w)

# get temporally subsampled frames for real/fake sequences
def get_skipped_frames(B_all, B, t_scales, n_frames_D):
    B_all = torch.cat([B_all.detach(), B], dim=1) if B_all is not None else B
    B_skipped = [None] * t_scales
    for s in range(t_scales):
        n_frames_Ds = n_frames_D ** s
        span = n_frames_Ds * (n_frames_D-1)
        n_groups = min(B_all.size()[1] - span, B.size()[1])
        if n_groups > 0:
            for t in range(0, n_groups, n_frames_D):
                skip = B_all[:, (-span-t-1):-t:n_frames_Ds].contiguous() if t != 0 else B_all[:, -span-1::n_frames_Ds].contiguous()
                B_skipped[s] = torch.cat([B_skipped[s], skip]) if B_skipped[s] is not None else skip
    max_prev_frames = n_frames_D ** (t_scales-1) * (n_frames_D-1)
    if B_all.size()[1] > max_prev_frames:
        B_all = B_all[:, -max_prev_frames:]
    return B_all, B_skipped

# get temporally subsampled frames for flows
def get_skipped_flows(flowNet, flow_ref_all, conf_ref_all, real_B, flow_ref, conf_ref, t_scales, n_frames_D):
    flow_ref_skipped, conf_ref_skipped = [None] * t_scales, [None] * t_scales
    flow_ref_all, flow = get_skipped_frames(flow_ref_all, flow_ref, 1, n_frames_D)
    conf_ref_all, conf = get_skipped_frames(conf_ref_all, conf_ref, 1, n_frames_D)
    if flow[0] is not None:
        flow_ref_skipped[0], conf_ref_skipped[0] = flow[0][:,1:], conf[0][:,1:]

    for s in range(1, t_scales):
        if real_B[s] is not None and real_B[s].size()[1] == n_frames_D:
            flow_ref_skipped[s], conf_ref_skipped[s] = flowNet(real_B[s][:,1:], real_B[s][:,:-1])
    return flow_ref_all, conf_ref_all, flow_ref_skipped, conf_ref_skipped

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if isinstance(image_tensor, torch.autograd.Variable):
        image_tensor = image_tensor.data
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor[0]
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * std + mean)  * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:,:,0]
    elif image_numpy.shape[2] == 2: # uv image case
        zeros = np.zeros((image_numpy.shape[0], image_numpy.shape[1], 1)).astype(int)
        image_numpy = np.concatenate([image_numpy, zeros], 2)
    return image_numpy.astype(imtype)

def tensor2flow(output, imtype=np.uint8):
    if isinstance(output, torch.autograd.Variable):
        output = output.data
    if len(output.size()) == 4:
        output = output[0]
    output = output.cpu().float().numpy()
    output = np.transpose(output, (1, 2, 0))
    #mag = np.max(np.sqrt(output[:,:,0]**2 + output[:,:,1]**2))
    #print(mag)
    hsv = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(output[..., 0], output[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def fit_ROI_in_frame(center, opt):
    center_w, center_h = center[0], center[1]
    center_h = torch.tensor(opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_h < opt.ROI_size // 2 else center_h
    center_w = torch.tensor(opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_w < opt.ROI_size // 2 else center_w
    center_h = torch.tensor(opt.loadSize - opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_h > opt.loadSize - opt.ROI_size // 2 else center_h
    center_w = torch.tensor(opt.loadSize - opt.ROI_size // 2, dtype=torch.int32).cuda(opt.gpu_ids[0]) if center_w > opt.loadSize - opt.ROI_size // 2 else center_w
    return (center_w, center_h)

def crop_ROI(img, center, ROI_size):
    return img[..., center[1] - ROI_size // 2:center[1] + ROI_size // 2,
                    center[0] - ROI_size // 2:center[0] + ROI_size // 2]

def get_ROI(tensors, centers, opt):
    real_A, real_B, fake_B = tensors
    # Extract region of interest around the center.
    real_A_ROI = []
    real_B_ROI = []
    fake_B_ROI = []
    for t in range(centers.shape[0]):
        center = fit_ROI_in_frame(centers[t], opt)
        real_A_ROI.append(crop_ROI(real_A[t], center, opt.ROI_size))
        real_B_ROI.append(crop_ROI(real_B[t], center, opt.ROI_size))
        fake_B_ROI.append(crop_ROI(fake_B[t], center, opt.ROI_size))
    real_A_ROI = torch.stack(real_A_ROI, dim=0)
    real_B_ROI = torch.stack(real_B_ROI, dim=0)
    fake_B_ROI = torch.stack(fake_B_ROI, dim=0)
    return real_A_ROI, real_B_ROI, fake_B_ROI

def draw_str(image, target, s):
    # Draw string for visualisation.
    x, y = target
    cv2.putText(image, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(image, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

def get_pixel_distance(rgb_frame, fake_frame, total_distance, total_pixels, nmfc_frame=None):
    # If NMFC frame is given, use it as a mask.
    mask = None
    if nmfc_frame is not None:
        mask = np.sum(nmfc_frame, axis=2)
        mask = (mask > (np.ones_like(mask) * 0.01)).astype(np.int32)
    # Sum rgb distance across pixels.
    error = abs(rgb_frame.astype(np.int32) - fake_frame.astype(np.int32))
    if mask is not None:
        distance = np.multiply(np.linalg.norm(error, axis=2), mask)
        n_pixels = mask.sum()
    else:
        distance = np.linalg.norm(error, axis=2)
        n_pixels = distance.shape[0] * distance.shape[1]
    sum_distance = distance.sum()
    total_distance += sum_distance
    total_pixels += n_pixels
    # Heatmap
    maximum = 50.0
    minimum = 0.0
    maxim = maximum * np.ones_like(distance)
    distance_trunc = np.minimum(distance, maxim)
    zeros = np.zeros_like(distance)
    ratio = 2 * (distance_trunc-minimum) / (maximum - minimum)
    b = np.maximum(zeros, 255*(1 - ratio))
    r = np.maximum(zeros, 255*(ratio - 1))
    g = 255 - b - r
    heatmap = np.stack([r, g, b], axis=2).astype(np.uint8)
    if nmfc_frame is not None:
        heatmap = np.multiply(heatmap, np.expand_dims(mask, axis=2)).astype(np.uint8)
    draw_str(heatmap, (20, 20), "%0.1f" % (sum_distance/n_pixels))
    return total_distance, total_pixels, heatmap
