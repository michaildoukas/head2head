# head2head

## Installation

- Clone the repository
```bash
git clone https://github.com/michaildoukas/head2head.git
cd head2head
```
### Build with Docker (recommended):

### Build with Conda:
(Requires cuda 9.2, vulkan)
- Create a conda environment, using the provided ```conda-env.txt``` file.
```bash
conda create --name head2headMP --file conda-env.txt
```
- Activate the conda environment.
```bash
conda activate head2headMP
```
- Install insightface and mxnet (used only for 3D face reconstruction):
```bash
pip install insightface mxnet-cu92mkl facenet-pytorch
```
