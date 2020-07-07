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
                        help='The dataset to download.')
    args = parser.parse_args()
    print('Download complete %s dataset\n' % args.dataset)
    if args.dataset == 'head2headDataset' or args.dataset == 'head2headDatasetv2':
        save_dir = os.path.join('datasets', args.dataset)
        if args.dataset == 'head2headDataset':
            links_list = [('dataset.zip', 'https://www.dropbox.com/s/424wm7cp2fa4o2o/dataset.zip?dl=1'),
                          ('original_videos.zip', 'https://www.dropbox.com/s/qzpfz47nwtfryad/original_videos.zip?dl=1')]
            for link in links_list:
                save_path = os.path.join(save_dir, link[0])
                if not os.path.exists(save_path):
                    bar = MyProgressBar('Downloading %s' % link[0])
                    wget.download(link[1], save_dir, bar=bar.get_bar)
                    print('\n')
                print('Unzipping file')
                unzip_file(save_path, save_dir)
        else:
            link = ('original_videos.zip', 'https://www.dropbox.com/s/5s3bqkvc4asppgd/original_videos.zip?dl=1')
            save_path = os.path.join(save_dir, link[0])
            if not os.path.exists(save_path):
                bar = MyProgressBar('Downloading %s' % link[0])
                wget.download(link[1], save_dir, bar=bar.get_bar)
                print('\n')
            print('Unzipping file')
            unzip_file(save_path, save_dir)
            dir = 'datasets/head2headDatasetv2'
            dataset_dir = 'datasets/head2headDatasetv2/dataset'
            download_paths_zip_parts = \
                ['https://www.dropbox.com/s/t2unzm9logbzg1e/dataset.zip?dl=1',
                 'https://www.dropbox.com/s/qgv61mnkhizmedv/dataset.z01?dl=1']
            for i, path in enumerate(download_paths_zip_parts):
                if not os.path.exists(os.path.join(dir, \
                                      path.split('/')[-1].split('?')[0])):
                    bar = MyProgressBar('Downloading dataset part %d/%d' % (i+1,
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
            print('Unzipping file, this might take some time...')
            unzip_file(dataset_dir + '_all.zip', dir)
    else:
        dir = 'datasets'
        dataset_dir = 'datasets/faceforensicspp'
        download_paths_zip_parts = \
            ['https://www.dropbox.com/s/upw2lvakm34pckz/faceforensicspp.zip?dl=1',
             'https://www.dropbox.com/s/9231mxrt6h9r8v5/faceforensicspp.z01?dl=1',
             'https://www.dropbox.com/s/5a0in2btxm6gc6e/faceforensicspp.z02?dl=1',
             'https://www.dropbox.com/s/8npxgw80o47v7ot/faceforensicspp.z03?dl=1',
             'https://www.dropbox.com/s/f4o57w3yxd9bc9e/faceforensicspp.z04?dl=1',
             'https://www.dropbox.com/s/hjk70q3k2xgr92d/faceforensicspp.z05?dl=1',
             'https://www.dropbox.com/s/uu9gkb567swi7z6/faceforensicspp.z06?dl=1',
             'https://www.dropbox.com/s/b0rhg2ya89erdiz/faceforensicspp.z07?dl=1',
             'https://www.dropbox.com/s/mzrygkqlvbecny8/faceforensicspp.z08?dl=1',
             'https://www.dropbox.com/s/1y6n0llhoajnfd1/faceforensicspp.z09?dl=1']
        for i, path in enumerate(download_paths_zip_parts):
            if not os.path.exists(os.path.join(dir, \
                                  path.split('/')[-1].split('?')[0])):
                bar = MyProgressBar('Downloading dataset part %d/%d' % (i+1,
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
