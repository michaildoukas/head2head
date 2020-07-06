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
    print('Downloading essential files')
    # Download urls (hardcoded)
    download_urls = \
        [('preprocessing', 'https://www.dropbox.com/s/iu0comdruy5ps2h/files.zip?dl=1'),
         ('preprocessing', 'https://www.dropbox.com/s/lgxigl5v6rmyq6g/models.zip?dl=1'),
         ('models/flownet2_pytorch', 'https://www.dropbox.com/s/qs0fb1itevjs886/FlowNet2_checkpoint.pth_faceflow.tar?dl=1'),
         ('models/flownet2_pytorch', 'https://www.dropbox.com/s/nmqeuv4nw3nfvag/FlowNet2_checkpoint.pth.tar?dl=1')]
    for i, url in enumerate(download_urls):
        fpath = os.path.join(url[0], url[1].split('/')[-1].split('?')[0])
        if not os.path.exists(fpath):
            bar = MyProgressBar('Downloading %s (%d/%d)' % (fpath, i+1,
                                len(download_urls)))
            wget.download(url[1], fpath, bar=bar.get_bar)
            print('\n')
            if fpath.endswith('.zip'):
                unzip_file(fpath, url[0])
        else:
            print('%s (%d/%d) exists!' % (fpath, i+1, len(download_urls)))
    print('DONE!')

if __name__ == "__main__":
    main()
