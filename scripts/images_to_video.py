import cv2
from PIL import Image
import numpy as np
import os
import sys
import argparse

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.pgm', '.PGM',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff',
    '.txt', '.json'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def write_video_to_file(video_file_path, video, mult_duration=1, n_last_frames_omit=0):
    image_size_h = video.shape[1]
    image_size_w = video.shape[2]
    writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (image_size_w, image_size_h), True)
    t = 0
    # mult_duration: how slower should the video be - was 3
    while t < video.shape[0] - n_last_frames_omit:
        for _ in range(mult_duration):
            writer.write(video[t,:,:,:])
        t += 1
    writer.release()
    print("Sample saved to: " + video_file_path)

def images_to_video(image_files_path, name):
    # Open images
    images = []
    image_names = [image_file for image_file in image_files_path if is_image_file(image_file) and name in image_file]
    image_names.sort()
    for image_name in image_names:
        images.append(np.array(Image.open(image_name)))
    video = np.array(images)
    return video[:,:,:,::-1]

def make_video_dataset(dir):
    images = []
    fnames = sorted(os.walk(dir))
    for fname in sorted(fnames):
        paths = []
        root = fname[0]
        for f in sorted(fname[2]):
            if is_image_file(f):
                paths.append(os.path.join(root, f))
        if len(paths) > 0:
            images.append(paths)
    # Find minimun sequence length and reduce all sequences to that.
    return images

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
    print('--- Create .mp4 video file from images --- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,
                        default='head2head_trudeau',
                        help='Name of model we used to generate the results.')
    parser.add_argument('--only_fake_frames', action='store_true',
                        help='Write only the fake frames to the video file.')
    args = parser.parse_args()
    print_args(parser, args)

    path = os.path.join('results', args.name)
    if not os.path.isdir(path):
        raise ValueError(path + ' path does not exist.')
    else:
        print('Converting results in %s to .mp4 videos.' % path)

    image_paths = make_video_dataset(path)

    for imgs in image_paths:
        save_path = os.path.join(os.path.dirname(imgs[0]),
                    os.path.dirname(imgs[0]).replace('/', '-') + '.mp4')
        if not os.path.exists(save_path):
            fake_video = images_to_video(imgs, '/fake_video')
            nmfc_video = images_to_video(imgs, '/nmfc_video')
            rgb_video = images_to_video(imgs, '/rgb_video')
            assert fake_video.shape[0] == nmfc_video.shape[0], 'Not correct number of image files.'
            assert rgb_video.shape[0] == nmfc_video.shape[0], 'Not correct number of image files.'
            print('Processing %d frames...' % fake_video.shape[0])
            if args.only_fake_frames:
                video_list = [fake_video]
            else:
                video_list = [rgb_video, nmfc_video, fake_video]
            final_video = np.concatenate(video_list, axis=2)
            write_video_to_file(save_path, final_video)

if __name__=='__main__':
    main()
