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
    save_dir = 'datasets/head2headDataset'
    links_list = [('dataset.zip', 'https://www.dropbox.com/s/saimhaftz27fjqt/dataset.zip?dl=1'),
                  ('original_videos.zip', 'https://www.dropbox.com/s/moh71pvtll9n9ye/original_videos.zip?dl=1')]
    for link in links_list:
        save_path = os.path.join(save_dir, link[0])
        if not os.path.exists(save_path):
            bar = MyProgressBar('Downloading %s' % link[0])
            wget.download(link[1], save_dir, bar=bar.get_bar)
            print('\n')
        print('Unzipping file')
        unzip_file(save_path, save_dir)
    print('DONE!')

if __name__ == "__main__":
    main()
