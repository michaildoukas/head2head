import zipfile
import wget
import os
import argparse

class MyProgressBar():
    def __init__(self, message):
        self.message = message

    def get_bar(self, current, total, width=80):
        print(self.message + ": %d%%" % (current / total * 100), end="\r")

def unzip_file(file_name, unzip_path):
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall(unzip_path)
    zip_ref.close()
    os.remove(file_name)

def main():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='head2headDataset',
                        choices=['head2headDataset', 'head2headDatasetv2', 'faceforensicspp'],
                        help='The dataset checkpoint to download. \
                        [head2headDataset|head2headDatasetv2|faceforensicspp]')
    args = parser.parse_args()
    print('Downloading trained models for all target identities in %s dataset\n' % args.dataset)
    save_dir = './'
    save_path = 'checkpoints.zip'
    if args.dataset == 'head2headDatasetv2':
        link = ('checkpoints.zip', 'https://www.dropbox.com/s/kmg1eaklr2agse9/checkpoints.zip?dl=1')
    elif args.dataset == 'faceforensicspp':
        link = ('checkpoints.zip', 'https://www.dropbox.com/s/gb4gzmjtypc7b4m/checkpoints.zip?dl=1')
    else:
        link = ('checkpoints.zip', 'https://www.dropbox.com/s/ti8nv0jeb3camcj/checkpoints.zip?dl=1')
    bar = MyProgressBar('Downloading %s' % link[0])
    wget.download(link[1], save_dir, bar=bar.get_bar)
    print('\n')
    print('Unzipping file')
    unzip_file(save_path, save_dir)
    print('DONE!')

if __name__ == "__main__":
    main()
