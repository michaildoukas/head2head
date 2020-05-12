import os
from skimage import io
import numpy as np
import dlib
import argparse
import collections
from tqdm import tqdm

IMG_EXTENSIONS = ['.png']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths_dict(dir):
    # Returns dict: {name: [path1, path2, ...], ...}
    image_files = {}
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname) and '/images/' in root:
                path = os.path.join(root, fname)
                seq_name = os.path.basename(root).split('_')[0]
                if seq_name not in image_files:
                    image_files[seq_name] = [path]
                else:
                    image_files[seq_name].append(path)
    # Sort paths for each sequence
    for k, v in image_files.items():
        image_files[k] = sorted(v)
    # Return directory sorted for keys (identity names)
    return collections.OrderedDict(sorted(image_files.items()))

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_landmarks(image_pths, landmarks):
    # Make dirs
    landmark_pths = [p.replace('/images/', '/landmarks68/') for p in image_pths]
    out_paths = set(os.path.dirname(landmark_pth) for landmark_pth in landmark_pths)
    for out_path in out_paths:
        mkdir(out_path)
    print('Saving results')
    for landmark, image_pth in tqdm(zip(landmarks, image_pths), total=len(image_pths)):
        landmark_file = os.path.splitext(image_pth.replace('/images/', '/landmarks68/'))[0] + '.txt'
        np.savetxt(landmark_file, landmark)

def dirs_exist(image_pths):
    nmfc_pths = [p.replace('/images/', '/landmarks68/') for p in image_pths]
    out_paths = set(os.path.dirname(nmfc_pth) for nmfc_pth in nmfc_pths)
    return all([os.path.exists(out_path) for out_path in out_paths])

def detect_landmarks(img_paths, detector, predictor):
    landmarks = []
    prev_points = None
    for i in tqdm(range(len(img_paths))):
        img = io.imread(img_paths[i])
        dets = detector(img, 1)
        if len(dets) > 0:
            shape = predictor(img, dets[0])
            points = np.empty([68, 2], dtype=int)
            for b in range(68):
                points[b,0] = shape.part(b).x
                points[b,1] = shape.part(b).y
            prev_points = points
            landmarks.append(points)
        else:
            print('No face detected,  using previous landmarks')
            landmarks.append(prev_points)
    return landmarks

def main():
    print('---------- 68 landmarks detector --------- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='head2headDataset', help='Path to the dataset directory.')
    args = parser.parse_args()

    predictor_path = 'preprocessing/files/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    args.dataset_path = os.path.join('datasets', args.dataset_name, 'dataset')

    images_dict = get_image_paths_dict(args.dataset_path)
    n_image_dirs = len(images_dict)
    print('Number of identities for landmark detection: %d \n' % n_image_dirs)

    # Iterate through the images_dict
    n_completed = 0
    for name, image_pths in images_dict.items():
        n_completed += 1
        if not dirs_exist(image_pths):
            landmarks = detect_landmarks(image_pths, detector, predictor)
            save_landmarks(image_pths, landmarks)
            print('(%d/%d) %s [SUCCESS]' % (n_completed, n_image_dirs, name))
        else:
            print('(%d/%d) %s already processed!' % (n_completed, n_image_dirs, name))

if __name__=='__main__':
    main()
