# Head2Head: Video-based Neural Head Synthesis

## Installation

Clone the repository
```bash
git clone https://github.com/michaildoukas/head2head.git
cd head2head
```

We provide two alternatives for installing head2head required packages:
- Build a Docker image (recommended, requires sudo privileges)
- Create a Conda environment (Requires CUDA 9.2 and Vulkan already installed)

#### Build a Docker image (option 1):
Install Docker and its dependencies:
```bash
sudo ./docker/ubuntu/xenial/vulkan-base/pre_docker_install.sh
```
Build docker image (Requires about 15 minutes):
```bash
sudo ./docker/ubuntu/xenial/vulkan-base/build.sh
```
Run container over the image:
```bash
sudo ./docker/ubuntu/xenial/vulkan-base/run.sh
```
Change to head2head directory (inside the container):
```bash
cd head2head
```

#### Create a Conda environment (option 2):
Create a conda environment, using the provided ```conda-env.txt``` file.
```bash
conda create --name head2head --file conda-env.txt
```
Activate the conda environment.
```bash
conda activate head2head
```
Install facenet-pytorch, insightface and mxnet with pip (inside the environment):
```bash
pip install insightface mxnet-cu92mkl facenet-pytorch
```

## head2head Dataset

#### Video data visualisation after face detection and cropping

We have trained and tested head2head on the seven target identities shown below:

![](imgs/head2headDataset_identities.gif)

#### Download head2head Dataset

- Link to the original videos videos, before ROI extraction: [\[original_videos.zip\]](https://www.dropbox.com/s/moh71pvtll9n9ye/original_videos.zip?dl=1). The YouTube urls, along with the start and stop timestamps are listed in ```datasets/head2headDataset/urls.txt``` file.
- Link to full dataset, with the extracted ROI frames and 3D reconstruction data (NMFCs, landmarks, expression, identity and camera parameters): [\[dataset.zip\]](https://www.dropbox.com/s/saimhaftz27fjqt/dataset.zip?dl=1)

Alternatively, you can download the head2head Dataset, running:

```bash
python scripts/download/download_dataset.py
```

It will be placed under ```datasets/head2headDataset```.

#### head2head Dataset structure
```
head2headDataset ----- original_videos
                   |
                   --- dataset ----- train ----- exp_coeffs (expression vectors)
                                 |           |
                                 --- test    --- id_coeffs (identity vector)
                                    (same    |
                                  structure) --- images (ROI RGB frames)
                                             |
                                             --- landmarks (5 facial landmarks)
                                             |
                                             --- misc (camera parameters - pose)
                                             |
                                             --- nmfcs (GAN conditional input)
```

## Create your Dataset

You can create your own dataset from .mp4 video files. For that **face detection** is applied first and a fixed bounding box is used to extract the ROI, around the face. Then, we perform **3D face reconstruction** and compute the NMFC images, one for each frame of the video.

#### Face detection (tracking)

In order to perform face detection and crop the facial region from a single .mp4 file or a directory with multiple files, run:

```bash
python preprocessing/detect.py --original_videos_path <videos_path> --default_split <split>
```

- ```<videos_path>``` is the path to the .mp4 file, or a directory of videos.

- ```<split>``` is the data split to place the file. It can be set to ```train``` (for videos-identities used as target) or ```test``` (for videos-identities used only as source).

#### Face reconstruction

Make sure you have downloaded the required models and files for 3D facial fitting, with:

```bash
python scripts/download_preprocessing_files.py
```

To perform 3D facial reconstruction and compute the NMFC images of all videos-identities in the dataset, please run:

```bash
python preprocessing/reconstruct.py
```

Please execute the command above each time you use the face detection script to add new identities (source or target) in the dataset.

## Train a head2head model

First, compile FlowNet2:
```bash
python scripts/compile_flownet2.py
```

In order to train a new person-specific model from scratch, use:

```bash
./scripts/train/train_head2head_on_target.sh <target_name>
```

where ```<target_name>``` is the name of the target identity, **which should have been already placed in the dataset** (after applying face detection and reconstruction on the original video file: ```<target_name>.mp4```).

## Test self reenactment

In self reenactment, the target person is also used as source. In this way we have access to the ground truth video, which provides a means to evaluate the performance of our model.

When adding a target identity to the dataset, during the face detection step, we leave a few frames out of the training set (about last 100 frames of target video), in order to use them for testing the model in a self reenactment scenario.

The following commands generates a video, using as driving input (source video) the kept out, test frames of ```<target_name>```:

```bash
./scripts/train/test_head2head_on_target.sh <target_name>
```

Synthesised videos are saved by default under the ```./results``` directory.

## Test head-to-head reenactment

For transferring the expressions and head pose from a source person, to a target person in our dataset, first we compute the NMFC frames that correspond to the source video, using the 3DMM identity coefficients computed from the target. For better quality, we adapt the mean Scale and Translation camera parameters of the source to the target.

Given a ```<source_name>``` and a ```<target_name>```, the NMFC sequence (conditional input to head2head model) is produced after running:

```bash
python preprocessing/reenact.py --source_id <source_name> --target_id <target_name> --split_s <source_split> --split_t <target_split>
```
were ```<source_split>``` is the split where the source video belongs (```test``` or ```train```) and ```<target_split>``` should be set to ```train```.

Then, we generate the synthetic video, with:
```bash
./scripts/train/test_reenactment_head2head_on_target.sh <target_name>
```

## Real-time head-to-head reenactment
