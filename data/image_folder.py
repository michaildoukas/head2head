import os
import random
from PIL import Image
import torch.utils.data as data

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.pgm', '.PGM',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff',
    '.txt', '.json'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def make_video_dataset(dir, target_name, source_name, max_n_sequences=None):
    images = []
    if dir:
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        fnames = sorted(os.walk(dir))
        for fname in sorted(fnames):
            paths = []
            root = fname[0]
            for f in sorted(fname[2]):
                names = os.path.basename(root).split('_')
                target = names[0]
                source = names[1] if source_name is not None else None
                if is_image_file(f):
                    if (target_name is None or target_name == target):
                        if (source_name is None or source_name == source):
                            paths.append(os.path.join(root, f))
            if len(paths) > 0:
                images.append(paths)
        if max_n_sequences is not None:
            images = images[:max_n_sequences]
    return images

def assert_valid_pairs(A_paths, B_paths):
    assert len(A_paths) > 0 and len(B_paths) > 0, 'No sequences found.'
    assert len(A_paths) == len(B_paths), 'Number of NMFC sequences different than RGB sequences.'
    for i in range(len(A_paths)):
        assert len(A_paths[i]) == len(B_paths[i]), 'Number of NMFC frames in sequence different than corresponding RGB frames.'

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
