import os
import cv2
import numpy as np
import pandas as pd
import scipy.signal
from PIL import Image
import torch
import argparse
from facenet_pytorch import MTCNN, extract_face
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm

VID_EXTENSIONS = ['.mp4']

def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VID_EXTENSIONS)

def tensor2npimage(image_tensor, imtype=np.uint8):
    # Tesnor in range [0,255]
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2npimage(image_tensor[i], imtype))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_images(images, name, split, start_i, is_last, args):
    if split == 'train' and is_last:
        n_parts = (len(images) + start_i) // args.train_seq_length
        assert n_parts - args.n_parts_test > 0, 'Number of test parts is more than available parts.'
        n_images_train = (n_parts - args.n_parts_test) * args.train_seq_length - start_i
        n_images_test = len(images) - n_images_train
    elif split == 'train':
        n_images_train = len(images)
        n_images_test = 0
    else:
        n_images_train = 0
        n_images_test = len(images)
    print('Saving images')
    for i in tqdm(range(len(images))):
        split_i = 'train' if i < n_images_train else 'test'
        n_frame = "{:06d}".format(i + start_i)
        part = "_{:06d}".format((i + start_i) // args.train_seq_length) if split == 'train' and i < n_images_train else ""
        save_dir = os.path.join(args.dataset_path, split_i, 'images', name + part)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_image(images[i], os.path.join(save_dir, n_frame + '.png'))

def get_video_paths_dict(dir):
    # Returns dict: {video_name: path, ...}
    if os.path.exists(dir) and is_video_file(dir):
        # If path to single .mp4 file was given directly.
        # If '_' in file name remove it.
        video_files = {os.path.splitext(os.path.basename(dir))[0].replace('_', '') : dir}
    else:
        video_files = {}
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_video_file(fname):
                    path = os.path.join(root, fname)
                    video_name = os.path.splitext(fname)[0]
                    # If part of video
                    if '_part_' in video_name:
                        video_name = video_name.split('_part_')[0]
                    video_name = video_name.replace('_', '')
                    if video_name not in video_files:
                        video_files[video_name] = [path]
                    else:
                        video_files[video_name].append(path)
    return collections.OrderedDict(sorted(video_files.items()))

def is_video_path_processed(name, split, args):
    first_part = '_000000' if split == 'train' else ''
    path = os.path.join(args.dataset_path, split, 'images', name + first_part)
    return os.path.isdir(path)

def read_mp4(mp4_path, args):
    reader = cv2.VideoCapture(mp4_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    images = []
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Reading %s' % mp4_path)
    for i in tqdm(range(n_frames)):
        _, image = reader.read()
        if image is None:
            break
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    reader.release()
    return images, fps

def check_boxes(boxes, img_size, args):
    # Check if there are None boxes. Fix them if only a few (like 5) are None
    not_detected_cases = 0
    for i in range(len(boxes)):
        if boxes[i] is None:
            not_detected_cases += 1
            boxes[i] = next((item for item in boxes[i+1:] if item is not None), boxes[i-1])
    if boxes[0] is None or not_detected_cases > args.None_threshold:
        print('Not enough boxes detected in video.')
        return False, None
    boxes = [box[0] for box in boxes]
    # Smoothen boxes
    old_boxes = np.array(boxes)
    if old_boxes.shape[0] <= args.window_length:
        print('Not enough boxes in video for savgol smoothing.')
        return False, None
    smooth_boxes = scipy.signal.savgol_filter(old_boxes, args.window_length, args.polyorder, axis=0)
    if np.any(smooth_boxes < 0):
        print('Negative box boundry detected in video.')
        return False, None
    # Check if detected faces are very far from each other. Check distances between all boxes.
    maxim_dst = 0
    for i in range(len(smooth_boxes)-1):
        for j in range(len(smooth_boxes)-1):
            dst = max(abs(smooth_boxes[i] - smooth_boxes[j])) / img_size
            if dst > maxim_dst:
                maxim_dst = dst
    if maxim_dst > args.dst_threshold:
         print('L_inf distance between bounding boxes %.4f larger than threshold' % maxim_dst)
         return False, None
    if args.keep_fixed_box:
        avg_box = np.median(smooth_boxes, axis=0)
        new_boxes = np.stack([avg_box] * smooth_boxes.shape[0], axis=0)
    else:
        new_boxes = smooth_boxes
    # Keep a fixed up-bottom, right-left offset and make boxes square.
    offset_w = np.mean(new_boxes[:,2] - new_boxes[:,0])
    offset_h = np.mean(new_boxes[:,3] - new_boxes[:,1])
    offset_dif = (offset_h - offset_w) / 2
    # width
    new_boxes[:,0] = new_boxes[:,2] - offset_w - offset_dif
    new_boxes[:,2] = new_boxes[:,2] + offset_dif
    # height - center a bit lower
    new_boxes[:,3] = new_boxes[:,3] + args.height_recentre * offset_h
    new_boxes[:,1] = new_boxes[:,3] - offset_h
    for i in range(new_boxes.shape[0]):
        boxes[i] = list(new_boxes[i,:])
    return True, boxes

def get_faces(detector, images, box, args):
    ret_faces = []
    all_boxes = []
    all_imgs = []
    if box is None:
        # Get bounding boxes
        print('Getting bounding boxes')
        for lb in tqdm(np.arange(0, len(images), args.mtcnn_batch_size)):
            imgs_pil = [Image.fromarray(image) for image in images[lb:lb+args.mtcnn_batch_size]]
            boxes, _, _ = detector.detect(imgs_pil, landmarks=True)
            all_boxes.extend(boxes)
            all_imgs.extend(imgs_pil)
        # Check if boxes are fine and do temporal smoothing.
        img_size = (all_imgs[0].size[0] + all_imgs[0].size[1]) / 2
        stat, all_boxes = check_boxes(all_boxes, img_size, args)
    else:
        all_imgs = [Image.fromarray(image) for image in images]
        stat, all_boxes = True, np.stack([box] * len(all_imgs), axis=0)
    # Crop face regions.
    if stat:
        print('Extracting faces')
        for img, box in tqdm(zip(all_imgs, all_boxes), total=len(all_boxes)):
            face = extract_face(img, box, args.cropped_image_size, args.margin)
            ret_faces.append(face)
    return stat, ret_faces, all_boxes[0]

def detect_and_save_faces(detector, name, mp4_paths, split, args):
    start_i = 0
    box = None
    for n, mp4_path in enumerate(mp4_paths):
        is_last = n == len(mp4_paths) - 1
        images, fps = read_mp4(mp4_path, args)
        stat, face_images, box = get_faces(detector, images, box, args)
        if stat:
            save_images(tensor2npimage(face_images), name, split, start_i, is_last, args)
            start_i += len(images)
    return stat

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
    print('-------------- Face detector -------------- \n')
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='Negative value to use CPU, or greater equal than zero for GPU id.')
    parser.add_argument('--original_videos_path', type=str, default='videos',
                        help='Path of video data dir.')
    parser.add_argument('--dataset_path', type=str, default='datasets/videos', help='Path to save dataset.')
    parser.add_argument('--mtcnn_batch_size', default=8, type=int, help='The number of frames for face detection.')
    parser.add_argument('--cropped_image_size', default=256, type=int, help='The size of frames after cropping the face.')
    parser.add_argument('--margin', default=100, type=int, help='.')
    parser.add_argument('--None_threshold', default=10, type=int, help='Max number of allowed None bounding boxes in a video.')
    parser.add_argument('--dst_threshold', default=0.3, type=float, help='Max L_inf distance between any bounding boxes in a video. (normalised by image size: (h+w)/2)')
    parser.add_argument('--window_length', default=99, type=int, help='savgol filter window length.')
    parser.add_argument('--polyorder', default=3, type=int, help='savgol filter polyorder.')
    parser.add_argument('--height_recentre', default=0.0, type=float, help='The amount of re-centring bounding boxes lower on the face.')
    parser.add_argument('--keep_fixed_box', action='store_true', default=True, help='Keep a fixed bounding box throughout the video.')
    parser.add_argument('--train_seq_length', default=50, type=int, help='The number of frames for each training sequence.')
    parser.add_argument('--n_parts_test', default=2, type=int, help='The number of sequences/parts left for test (self-reenactment)')
    parser.add_argument('--default_split', default='train', choices=['train', 'test'], type=str, help='The default split for data [train|test]')

    args = parser.parse_args()
    print_args(parser, args)

    # Figure out the device
    gpu_id = int(args.gpu_id)
    if gpu_id < 0:
        device = 'cpu'
    elif torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            device = 'cuda:0'
        else:
            device = 'cuda:' + str(gpu_id)
    else:
        print('GPU device not available. Exit')
        exit(0)

    # Store video paths in dictionary.
    mp4_paths_dict = get_video_paths_dict(args.original_videos_path)
    n_mp4s = len(mp4_paths_dict)
    print('Number of videos to process: %d \n' % n_mp4s)

    # Initialize the MTCNN face  detector.
    detector = MTCNN(image_size=args.cropped_image_size, margin=args.margin, post_process=False, device=device)

    # Run detection
    n_completed = 0
    for name, path in mp4_paths_dict.items():
        n_completed += 1
        if not is_video_path_processed(name, args.default_split, args):
            success = detect_and_save_faces(detector, name, path, args.default_split, args)
            if success:
                print('(%d/%d) %s (%s file) [SUCCESS]' % (n_completed, n_mp4s, path[0], args.default_split))
            else:
                print('(%d/%d) %s (%s file) [FAILED]' % (n_completed, n_mp4s, path[0], args.default_split))
        else:
            print('(%d/%d) %s (%s file) already processed!' % (n_completed, n_mp4s, path[0], args.default_split))

if __name__ == "__main__":
    main()
