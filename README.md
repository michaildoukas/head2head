# Head2Head: Video-based Neural Head Synthesis

| ![](imgs/head2head_demo.gif) |
|:--:|
| *Transferring head pose, facial expressions and eye gaze from a source video to a target identity* |

| ![imgs/face_reenactment_demo.gif](imgs/face_reenactment_demo.gif) | ![imgs/head_reenactment_demo.gif](imgs/head_reenactment_demo.gif) |
|:--:|:--:|
| *Simple face reenactment (facial expression transfer)* | *Complete head reenactment (pose, expression, eyes transfer)* |

## Installation

Clone the repository
```bash
git clone https://github.com/michaildoukas/head2head.git
cd head2head
```

We provide two alternatives for installing Head2Head required packages:
- Build a Docker image (Recommended, requires sudo privileges)
- Create a Conda environment (Requires python 3.7, CUDA 9.2 and Vulkan already installed)

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
Activate the environment.
```bash
conda activate head2head
```
Install dlib, facenet-pytorch, insightface and mxnet with pip (inside the environment):
```bash
pip install dlib insightface mxnet-cu92 facenet-pytorch
```

#### Download essential files

Make sure you have downloaded the required models and files for landmark detection, face reconstruction and FlowNet2 checkpoints, with:

```bash
python scripts/download_files.py
```

## Head2Head dataset

#### Video data visualisation after face detection and cropping

We have trained and tested Head2Head on the seven target identities shown below:

![](imgs/head2headDataset_identities.gif)

#### Download Head2Head Dataset

- Link to the seven original video files, before ROI extraction: [\[original_videos.zip\]](https://www.dropbox.com/s/qzpfz47nwtfryad/original_videos.zip?dl=1). The corresponding YouTube urls, along with the start and stop timestamps are listed in ```datasets/head2headDataset/urls.txt``` file.
- Link to full dataset, with the extracted ROI frames and 3D reconstruction data (NMFCs, landmarks, expression, identity and camera parameters): [\[dataset.zip\]](https://www.dropbox.com/s/424wm7cp2fa4o2o/dataset.zip?dl=1)

Alternatively, you can download Head2Head dataset, by running:

```bash
python scripts/download_dataset.py
```

It will be placed under ```datasets/head2headDataset```.

#### Head2Head Dataset structure

We split the original video of each identity into one training and one test sequence. We place about one third of the total number of frames in the test split. In this way, we are able to use these frames as ground truth, when testing the model in a self reenactment scenario.

```
head2headDataset ----- original_videos
                   |
                   --- dataset ----- train ----- exp_coeffs (expression vectors)
                                 |           |
                                 --- test    --- id_coeffs (identity vector)
                                    (same    |
                                  structure) --- images (ROI RGB frames)
                                             |
                                             --- landmarks70 (68 + 2 facial landmarks)
                                             |
                                             --- misc (camera parameters - pose)
                                             |
                                             --- nmfcs (GAN conditional input)
```

#### Head2Head Dataset version 2

We have added 7 new identities, with longer training video footage (10 mins +). Download via the links: [\[original_videos.zip\]](https://www.dropbox.com/s/7c8lci8c8b8pli7/original_videos.zip?dl=1), [\[dataset.zip\]](https://www.dropbox.com/s/kcdyoe85cob97lt/dataset.zip?dl=1), or by running:

```bash
python scripts/download_dataset.py --dataset head2headDatasetv2
```

## Create your Dataset

You can create your own dataset from .mp4 video files. For that, first we do **face detection**, which returns a fixed bounding box that is used to extract the ROI, around the face. Then, we perform **3D face reconstruction** and compute the NMFC images, one for each frame of the video.

#### Face detection (tracking)

In order to perform face detection and crop the facial region from a single .mp4 file or a directory with multiple files, run:

```bash
python preprocessing/detect.py --original_videos_path <videos_path> --dataset_name <dataset_name> --split <split>
```

- ```<videos_path>``` is the path to the original .mp4 file, or a directory of .mp4 files. (default: ```datasets/head2headDataset/original_videos```)

- ```<dataset_name>``` is the name to be given to the dataset. (default: ```head2headDataset```)

- ```<split>``` is the data split to place the file(s). If set to ```train```, the videos-identities can be used as target, but the last one third of the frames is placed in the test set, enabling self reenactment experiments. When set to ```test```, the videos-identities can be used only as source and no frames are placed in the training set. (default: ```train```)

#### 68 + 2 facial landmarks detection

```bash
python preprocessing/detect_landmarks70.py --dataset_name <dataset_name>
```

#### 3D face reconstruction

To perform 3D facial reconstruction and compute the NMFC images of all videos-identities in the dataset, run:

```bash
python preprocessing/reconstruct.py --dataset_name <dataset_name>
```

Please execute the two commands above (landmark detection and face reconstruction) each time you use the face detection script to add new identities in the ```<dataset_name>``` dataset.

## Train a head2head model

First, compile FlowNet2:
```bash
python scripts/compile_flownet2.py
```

In order to train a new person-specific model from scratch, use:

```bash
./scripts/train/train_on_target.sh <target_name> <dataset_name>
```

where ```<target_name>``` is the name of the target identity, which should have been already placed in the ```<dataset_name>``` dataset, after processing the original video file: ```<target_name>.mp4```.

## Test self reenactment

In self reenactment, the target person is also used as source. In this way we have access to the ground truth video, which provides a means to evaluate the performance of our model.

The following commands generates a video, using as driving input (source video) the kept out, test frames of ```<target_name>```:

```bash
./scripts/test/test_self_reenactment_on_target.sh <target_name> <dataset_name>
```

Synthesised videos are saved under the ```./results``` directory.

## Test source-to-target head reenactment

For transferring the expressions and head pose from a source person, to a target person in our dataset, first we compute the NMFC frames that correspond to the source video, using the 3DMM identity coefficients computed from the target. For better quality, we adapt the mean Scale and Translation camera parameters of the source to the target. Then, we generate the synthetic video, using these NMFC frames as conditional input.

Given a ```<source_name>``` and a ```<target_name>``` from dataset ```<dataset_name>```, head reenactment results are generated after running:
```bash
./scripts/test/test_reenactment_from_source_to_target.sh <source_name> <target_name> <dataset_name>
```

## Test source-to-target face reenactment

Instead of transferring the head pose from a source video, we can perform simple face reenactment, by keeping the original pose of the target video, and using only the expressions (inner facial movements) of the source.

For a ```<source_name>``` and a ```<target_name>``` from dataset ```<dataset_name>```, face reenactment results are generated after running:
```bash
./scripts/test/test_face_reenactment_from_source_to_target.sh <source_name> <target_name> <dataset_name>
```

## Real-time reenactment

TODO

## Pre-training on FaceForensic++ Dataset

In order to increase the generative performance of head2head in very short target videos, we can pre-train a model on the a multi-person dataset, such as FaceForensic++, and then fine-tune it on a new target video-identity. You can download a processed version of the original 1000 videos of FaceForensic++ with complete NMFC annotations (requires ~100 GBs of free disk space), with:

```bash
python scripts/download_dataset.py --dataset faceforensicspp
```
Then, train head2head on this multi-person dataset:
```bash
./scripts/train/train_faceforensicspp.sh
```
Finally, fine-tune on ```<target_name>``` from ```<dataset_name>```:
```bash
./scripts/train/finetune_on_target.sh <target_name> <dataset_name>
```
