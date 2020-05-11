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
    print('Downloading models and files for preprocessing \n')
    # Download urls (hardcoded)
    download_urls = \
        ['https://www.dropbox.com/s/vzr7snb82n42mfw/files.zip?dl=1',
         'https://www.dropbox.com/s/uzqz2p9fw3ps7pk/models.zip?dl=1']
    # Save dir (hardcoded)
    dir = 'preprocessing'
    for i, path in enumerate(download_urls):
        fpath = os.path.join(dir, path.split('/')[-1].split('?')[0])
        if not os.path.exists(fpath):
            bar = MyProgressBar('Downloading file %d/%d' % (i+1,
                                len(download_urls)))
            wget.download(path, fpath, bar=bar.get_bar)
            print('\n')
            print('Unzipping file...')
            unzip_file(fpath, dir)
    print('DONE!')

if __name__ == "__main__":
    main()
