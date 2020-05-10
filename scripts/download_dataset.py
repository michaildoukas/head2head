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
    print('Download complete head2head dataset\n')
    # Download paths (hardcoded)
    dir = 'datasets/head2headDataset'
    dataset_dir = 'datasets/head2headDataset/dataset'
    download_paths_zip_parts = \
        ['https://www.dropbox.com/s/hqnp7e6ee1l7nrk/dataset.zip?dl=1']
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    for i, path in enumerate(download_paths_zip_parts):
        if not os.path.exists(os.path.join(dir, \
                              path.split('/')[-1].split('?')[0])):
            bar = MyProgressBar('Downloading part %d/%d' % (i+1,
                                len(download_paths_zip_parts)))
            wget.download(path, dir, bar=bar.get_bar)
            print('\n')
    print('Merging parts into single .zip file...')
    os.system('zip -F ' + dataset_dir + '.zip \
               --out ' + dataset_dir + '_all.zip')
    print('Deleting parts...')
    for i, path in enumerate(download_paths_zip_parts):
        file_p  = os.path.join(dir, path.split('/')[-1].split('?')[0])
        if os.path.exists(file_p):
            os.remove(file_p)
    print('Unzipping file, this might take several minutes...')
    unzip_file(dataset_dir + '_all.zip', dir)
    print('DONE!')

if __name__ == "__main__":
    main()
