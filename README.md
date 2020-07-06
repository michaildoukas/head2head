## Head2Head: Video-based Neural Head Synthesis & Head2Head++: Deep Facial Attributes Re-Targeting

PyTorch implementation for Head2Head and Head2Head++. It can be used to fully transfer the head pose, facial expression and eye movements from a source video to a target identity.

![imgs/head2head.gif](imgs/head2head.gif)

> **Head2Head: Video-based Neural Head Synthesis**<br>
> [Mohammad Rami Koujan]()\*, [Michail Christos Doukas]()\*, [Anastasios Roussos](), [Stefanos Zafeiriou]()<br>
> In 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2020)<br>
> (* equal contribution)<br>
>
> Paper: https://arxiv.org/abs/2005.10954<br>
> Video Demo: https://youtu.be/RCvVMF5cVeY<br>

> **Head2Head++: Deep Facial Attributes Re-Targeting**<br>
>  [Michail Christos Doukas]()\*, [Mohammad Rami Koujan]()\*, [Anastasios Roussos](), [Viktoriia Sharmanska]() [Stefanos Zafeiriou]()<br>
> Submitted to the IEEE Transactions on Biometrics, Behavior, and Identity Science (TBIOM) journal.<br>
> (* equal contribution)<br>
>
> Paper: https://arxiv.org/abs/2006.10199<br>
> Video Demo: https://youtu.be/BhpRjjCcmJE<br>


## Reenactment Examples

| ![imgs/face_reenactment_demo.gif](imgs/face_reenactment_demo.gif) | ![imgs/head_reenactment_demo.gif](imgs/head_reenactment_demo.gif) |
|:--:|:--:|
| *Simple face reenactment (facial expression transfer)* | *Full head reenactment (pose, expression, eyes transfer)* |

## Installation

Clone the repository
```bash
git clone https://github.com/michaildoukas/head2head.git
cd head2head
```

We provide two alternatives for installing Head2Head required packages:
- Create a Conda environment (Requires python 3.7, CUDA 9.2 and Vulkan already installed)
- Build a Docker image (Recommended, requires sudo privileges)

#### (option 1) Create a Conda environment:
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

#### (option 2) Build a Docker image:
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

#### Compile FlowNet2

```bash
python scripts/compile_flownet2.py
```

If you are using docker, run the command above each time you run the container.

#### Download essential files

Make sure you have downloaded the required models and files for landmark detection, face reconstruction and FlowNet2 checkpoints, with:

```bash
python scripts/download_files.py
```

#### Acquiring the LSFM Models
In case you want to use your own source or target videos, you need to acquire the LSFM model files ```all_all_all.mat``` and ```lsfm_exp_30.dat``` and place them under the ```preprocessing/files``` directory. These files are essential for the 3D face reconstruction stage. For full terms and conditions, and to request access to the models, please visit the [LSFM website](http://ibug.doc.ic.ac.uk/resources/lsfm/). For more details on models, see [Large Scale Facial Model (LSFM)](https://github.com/menpo/lsfm).

## Head2Head dataset

#### Video data visualisation after face detection and cropping

We have trained and tested Head2Head on the seven target identities (Turnbull, Obama, Putin, Merkel, Trudeau, Biden, May) shown below:

![](imgs/head2headDataset_identities.png)

#### Download Head2Head Dataset

- Link to the seven original video files, before ROI extraction: [\[original_videos.zip\]](https://www.dropbox.com/s/qzpfz47nwtfryad/original_videos.zip?dl=1). The corresponding YouTube urls, along with the start and stop timestamps are listed in ```datasets/head2headDataset/urls.txt``` file.
- Link to full dataset, with the extracted ROI frames and 3D reconstruction data (NMFCs, landmarks, expression, identity and camera parameters): [\[dataset.zip\]](https://www.dropbox.com/s/424wm7cp2fa4o2o/dataset.zip?dl=1)

Alternatively, you can download Head2Head dataset, by running:

```bash
python scripts/download_dataset.py
```

You can download the fine-tuned models (checkpoints) for all seven target identities [here](https://www.dropbox.com/s/ti8nv0jeb3camcj/checkpoints.zip?dl=1), or with:

```bash
python scripts/download_checkpoints.py
```

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

## Head2Head++ Dataset

![](imgs/head2headDatasetv2_identities.png)

We have added eight new identities, with longer training video footage ( > 10 mins). Please download this dataset via the links: [\[original_videos.zip\]](https://www.dropbox.com/s/5s3bqkvc4asppgd/original_videos.zip?dl=1), [\[dataset.zip\]](https://www.dropbox.com/s/t2unzm9logbzg1e/dataset.zip?dl=1), or by running:

```bash
python scripts/download_dataset.py --dataset head2headDatasetv2
```

You can also download the trained models (checkpoints) for all eight target identities [here](https://www.dropbox.com/s/kmg1eaklr2agse9/checkpoints.zip?dl=1), or with:

```bash
python scripts/download_checkpoints.py --dataset head2headDatasetv2
```

## Create your Dataset

You can create your own dataset from .mp4 video files. For that, first do **face detection**, which returns a fixed bounding box that is used to extract the ROI, around the face. Then, perform **3D face reconstruction** and compute the NMFC images, one for each frame of the video. Finally, run **facial landmark localisation** to get the eye movements.

#### Face detection

In order to perform face detection and crop the facial region from a single .mp4 file or a directory with multiple files, run:

```bash
python preprocessing/detect.py --original_videos_path <videos_path> --dataset_name <dataset_name> --split <split>
```

- ```<videos_path>``` is the path to the original .mp4 file, or a directory of .mp4 files. (default: ```datasets/head2headDataset/original_videos```)

- ```<dataset_name>``` is the name to be given to the dataset. (default: ```head2headDataset```)

- ```<split>``` is the data split to place the file(s). If set to ```train```, the videos-identities can be used as target, but the last one third of the frames is placed in the test set, enabling self reenactment experiments. When set to ```test```, the videos-identities can be used only as source and no frames are placed in the training set. (default: ```train```)

#### 3D face reconstruction (requires LSFM files acquisition)

To perform 3D facial reconstruction and compute the NMFC images of all videos/identities in the dataset, run:

```bash
python preprocessing/reconstruct.py --dataset_name <dataset_name>
```

To extract facial landmarks (and eye pupils), run:

```bash
python preprocessing/detect_landmarks70.py --dataset_name <dataset_name>
```

Execute the commands above each time you use the face detection script to add new identity to ```<dataset_name>``` dataset.

## Train a head2head model

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

## Test head reenactment

For transferring the expressions and head pose from a source person, to a target person in our dataset, first we compute the NMFC frames that correspond to the source video, using the 3DMM identity coefficients computed from the target. For better quality, we adapt the mean Scale and Translation camera parameters of the source to the target. Then, we generate the synthetic video, using these NMFC frames as conditional input.

Given a ```<source_name>``` and a ```<target_name>``` from dataset ```<dataset_name>```, head reenactment results are generated after running:
```bash
./scripts/test/test_head_reenactment_from_source_to_target.sh <source_name> <target_name> <dataset_name>
```

## Test face reenactment

Instead of transferring the head pose from a source video, we can perform simple face reenactment, by keeping the original pose of the target video, and using only the expressions (inner facial movements) of the source.

For a ```<source_name>``` and a ```<target_name>``` from dataset ```<dataset_name>```, face reenactment results are generated after running:
```bash
./scripts/test/test_face_reenactment_from_source_to_target.sh <source_name> <target_name> <dataset_name>
```

## Reenactment demo

Nearly real-time demo using your camera:

```bash
./scripts/demo/run_demo_on_target.sh <target_name> <dataset_name>
```

## Pre-training on FaceForensic++ Dataset

In order to increase the generative performance of head2head in short target videos, we can pre-train a model on the a multi-person dataset, such as FaceForensic++, and then fine-tune it on a new target video-identity. You can download a processed version of the 1000 real videos of FaceForensic++ with complete NMFC annotations (requires ~100 GBs of free disk space), with:

```bash
python scripts/download_dataset.py --dataset faceforensicspp
```
and then train head2head on this multi-person dataset
```bash
./scripts/train/train_on_faceforensicspp.sh
```
Alternatively download the trained checkpoint [here](https://www.dropbox.com/s/gb4gzmjtypc7b4m/checkpoints.zip?dl=1), or by running:
```bash
python scripts/download_checkpoints.py --dataset faceforensicspp
```
Finally, fine-tune a model on ```<target_name>``` from ```<dataset_name>``` dataset:
```bash
./scripts/train/finetune_on_target.sh <target_name> <dataset_name>
```
Perform head reenactment:
```bash
./scripts/test/test_finetuned_head_reenactment_from_source_to_target.sh <source_name> <target_name> <dataset_name>
```
## Citation

If you use this code, please cite our Head2Head paper.

```
@INPROCEEDINGS {head2head2020,
author = {M. Koujan and M. Doukas and A. Roussos and S. Zafeiriou},
booktitle = {2020 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2020) (FG)},
title = {Head2Head: Video-Based Neural Head Synthesis},
year = {2020},
volume = {},
issn = {},
pages = {319-326},
keywords = {},
doi = {10.1109/FG47880.2020.00048},
url = {https://doi.ieeecomputersociety.org/10.1109/FG47880.2020.00048},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {may}
}
```

### Additional notes
This code borrows from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) and [vid2vid](https://github.com/NVIDIA/vid2vid).
