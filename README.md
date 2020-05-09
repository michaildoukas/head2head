# head2head

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

## Create a Dataset

You can create your dataset from .mp4 video files. For that **face detection** is applied first and a fixed bounding box is used to extract the ROI, around the face. Then, we perform **3D face reconstruction** and compute the NMFC images, one for each frame of the video.

#### Face detection (tracking)

In order to perform face detection and crop the facial region from a single .mp4 file or a directory with multiple files, run:

```bash
python preprocessing/detect.py --original_videos_path <videos_path> --default_split <split>
```

- ```<videos_path>``` is the path to the .mp4 file, or a directory of videos. The default location of video files is under ```./videos``` directory.

- ```<split>``` is the data split to place the file. It can be set to ```train``` (for target videos-identities) or ```test``` (for source videos-identities).

#### Face reconstruction

Make sure you have downloaded the required models and files for 3D facial fitting, with:

```bash
python scripts/download_preprocessing_files.py
```

To perform 3D facial reconstruction and compute the NMFC images of all videos-identities in the dataset, please run:

```bash
python preprocessing/reconstruct.py
```

The default dataset path is ```./datasets/videos```. Please run the command above each time you use the face detection script to add new identity(ies) (source or target) in the dataset.

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

## Test (self-reenactment)

When adding a target identity to the dataset, during the face detection step, we leave a few frames out of the training set (about last 100 frames of target video), in order to use them for testing the model in a self-reenactment scenario.

The following commands generates a video, using as driving input (source) the kept out test frames of ```<target_name>```:

```bash
./scripts/train/test_head2head_on_target.sh <target_name>
```

Synthesised videos are saved by default under the ```./results``` directory.

## Head to head reenactment (TODO)
