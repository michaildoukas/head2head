# edowson/vulkan-base/cudagl:$BUILD_VERSION-$CUDA_MAJOR_VERSION-xenial

ARG REPOSITORY
ARG TAG
FROM ${REPOSITORY}:${TAG}

# setup environment variables
ENV container docker
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}display,graphics,utility

# set the locale
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8

# Install packages (TODO:remove)
#RUN apt-get update && apt-get install -y rsync htop git openssh-server
#RUN apt-get update && apt-get install -y software-properties-common

# install packages
RUN apt-get update \
    && apt-get install -q -y \
    dirmngr \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# setup keys: llvm
ARG LLVM_VERSION
RUN if [ -n "$LLVM_VERSION" ]; then \
     echo "importing gpg keys for llvm apt repository" ;\
     apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 6084F3CF814B57C1CF12EFD515CF4D18AF4F7421 ;\
    fi

# setup sources.list
RUN echo "deb-src http://us.archive.ubuntu.com/ubuntu/ $(lsb_release -cs) main restricted \n\
deb-src http://us.archive.ubuntu.com/ubuntu/ $(lsb_release -cs)-updates main restricted \n\
deb-src http://us.archive.ubuntu.com/ubuntu/ $(lsb_release -cs)-backports main restricted universe multiverse \n\
deb-src http://security.ubuntu.com/ubuntu $(lsb_release -cs)-security main restricted" \
    > /etc/apt/sources.list.d/official-source-repositories.list \
    && if [ -n "$LLVM_VERSION" ]; then \
        echo "setting up sources.list for llvm-toolchain-$LLVM_VERSION" ;\
        echo "deb http://apt.llvm.org/`lsb_release -sc`/ llvm-toolchain-`lsb_release -sc`-$LLVM_VERSION main" \
        > "/etc/apt/sources.list.d/llvm-toolchain-$LLVM_VERSION.list" ;\
       fi

# install build tools
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TERM=linux apt-get install --no-install-recommends -q -y \
    apt-transport-https \
    apt-utils \
    bash-completion \
    build-essential \
    bzip2 \
    ca-certificates \
    curl \
    gconf2 \
    gconf-service \
    gdb \
    git-core \
    git-gui \
    gvfs-bin \
    inetutils-ping \
    nano \
    net-tools \
    pkg-config \
    pulseaudio-utils \
    rsync \
    shared-mime-info \
    software-properties-common \
    sudo \
    tzdata \
    unzip \
    wget \
    xdg-user-dirs \
    xdg-utils \
    x11-xserver-utils \
    zip \
    # install git-lfs
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && DEBIAN_FRONTEND=noninteractive TERM=linux apt-get install --no-install-recommends -q -y \
    git-lfs \
    # install llvm-toolchain
    && if [ -n "$LLVM_VERSION" ]; then \
        echo "installing llvm-toolchain-$LLVM_VERSION" ;\
        # install llvm-toolchain
        DEBIAN_FRONTEND=noninteractive TERM=linux apt-get install --no-install-recommends -q -y \
        clang-$LLVM_VERSION \
        clang-format-$LLVM_VERSION \
        lld-$LLVM_VERSION \
        lldb-$LLVM_VERSION \
        llvm-$LLVM_VERSION \
        llvm-$LLVM_VERSION-dev ;\
        # configure update update-alternatives \
        update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-$LLVM_VERSION 100 ;\
        update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-$LLVM_VERSION   100 ;\
        update-alternatives --install /usr/bin/ld.lld  ld.lld  /usr/bin/ld.lld-$LLVM_VERSION  100 ;\
        update-alternatives --install /usr/bin/lldb    lldb    /usr/bin/lldb-$LLVM_VERSION    100 ;\
       else \
        echo "llvm-toolchain not specified, not installing llvm." ;\
       fi \
    # perform dist-upgrade: dist-upgrade in addition to performing the function of upgrade,
    # also intelligently handles changing dependencies with new versions of packages
    && DEBIAN_FRONTEND=noninteractive TERM=linux apt-get dist-upgrade --no-install-recommends -q -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# install required libraries
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TERM=linux apt-get install -q -y \
    libasound2-dev \
    libglm-dev \
    libmirclient-dev \
    libpciaccess0 \
    libpng-dev \
    libwayland-dev \
    libxcb-ewmh-dev \
    libxcb-dri3-0 \
    libxcb-dri3-dev \
    libxcb-glx0 \
    libxcb-present0 \
    libxcb-keysyms1-dev \
    libx11-dev \
    libxrandr-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Install python3.7
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.7

# Set python3.7 to default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# Make python point to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install pip
RUN apt-get install python3-pip -y
RUN pip3 install --upgrade pip

# install cudnn libraries
ARG CUDA_MAJOR_VERSION
ARG CUDNN_VERSION
ENV CUDA_MAJOR_VERSION $CUDA_MAJOR_VERSION
ENV CUDNN_VERSION $CUDNN_VERSION
LABEL com.nvidia.cudnn.version="$CUDNN_VERSION"
RUN CUDA_MAJOR_VERSION=`echo $CUDA_VERSION | cut -d. -f1` \
    && CUDA_MINOR_VERSION=`echo $CUDA_VERSION | cut -d. -f2` \
    && CUDNN_MAJOR_VERSION=`echo $CUDNN_VERSION | cut -d. -f1` \
    && echo "CUDA_VERSION=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION" \
    && echo "CUDNN_VERSION=$CUDNN_VERSION" \
    && echo "CUDNN_MAJOR_VERSION=$CUDNN_MAJOR_VERSION" \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive TERM=linux apt-get install -q -y \
    libcudnn$CUDNN_MAJOR_VERSION=$CUDNN_VERSION-1+cuda$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION \
    libcudnn$CUDNN_MAJOR_VERSION-dev=$CUDNN_VERSION-1+cuda$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION \
    && apt-mark hold libcudnn$CUDNN_MAJOR_VERSION \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# install cmake
ARG CMAKE_VERSION=3.14
RUN wget -qO- "https://cmake.org/files/v$CMAKE_VERSION/cmake-$CMAKE_VERSION.0-Linux-x86_64.tar.gz" | \
  tar --strip-components=1 -xz -C /usr/local

# install vulkan sdk
ARG BUILD_VERSION
ARG VULKAN_SDK_VERSION
ENV VULKAN_SDK_VERSION=$VULKAN_SDK_VERSION
RUN echo "downloading Vulkan SDK $VULKAN_SDK_VERSION" \
    && wget -q --show-progress --progress=bar:force:noscroll https://sdk.lunarg.com/sdk/download/$VULKAN_SDK_VERSION/linux/vulkansdk-linux-$VULKAN_SDK_VERSION.tar.gz?Human=true -O /tmp/vulkansdk-linux-x86_64-$VULKAN_SDK_VERSION.tar.gz \
    && echo "installing Vulkan SDK $VULKAN_SDK_VERSION" \
    && mkdir -p /opt/vulkan \
    && tar -xf /tmp/vulkansdk-linux-x86_64-$VULKAN_SDK_VERSION.tar.gz -C /opt/vulkan \
    && rm /tmp/vulkansdk-linux-x86_64-$VULKAN_SDK_VERSION.tar.gz

# install nvidia driver
ARG NVIDIA_DRIVER_VERSION
ENV NVIDIA_DRIVER_VERSION $NVIDIA_DRIVER_VERSION
RUN wget -q --show-progress --progress=bar:force:noscroll http://us.download.nvidia.com/XFree86/Linux-x86_64/$NVIDIA_DRIVER_VERSION/NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION.run -O /tmp/NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION.run \
    && cd /tmp \
    && sh NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION.run --extract-only \
    && cp /tmp/NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION/libnvidia-cbl.so.$NVIDIA_DRIVER_VERSION /usr/lib/x86_64-linux-gnu/ \
    && cp /tmp/NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION/libnvidia-glvkspirv.so.$NVIDIA_DRIVER_VERSION /usr/lib/x86_64-linux-gnu/ \
    && cp /tmp/NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION/libnvidia-rtcore.so.$NVIDIA_DRIVER_VERSION /usr/lib/x86_64-linux-gnu/ \
    && rm -rf /tmp/NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION /tmp/NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION.run

# install packages and libraries for jetbrains ide
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TERM=linux apt-get install --no-install-recommends -q -y \
    # java jdk \
    openjdk-8-jdk \
    # gtk support \
    libcanberra-gtk-module \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# create user
ARG USER
ARG UID

ENV HOME /home/$USER
RUN adduser $USER --uid $UID --disabled-password --gecos "" \
    && usermod -aG audio,video $USER \
    && echo "$USER ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# copy configuration files
COPY docker/ubuntu/xenial/vulkan-base/config/etc/asound.conf /etc/asound.conf
COPY docker/ubuntu/xenial/vulkan-base/config/etc/pulse/client.conf /etc/pulse/client.conf
COPY docker/ubuntu/xenial/vulkan-base/config/etc/resolv.conf /etc/resolv.conf

# setup nvidia opengl and vulkan driver icds
#COPY docker/ubuntu/xenial/vulkan-base/config/vendor/egl_nvidia_icd.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY docker/ubuntu/xenial/vulkan-base/config/vendor/glx_nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json

# switch to non-root user
USER $USER

# set the working directory
WORKDIR $HOME

# install conda
ARG CONDA_PYTHON_VERSION
ARG CONDA_BASE_PACKAGE
ARG CONDA_VERSION
ENV CONDA_PYTHON_VERSION $CONDA_PYTHON_VERSION
ENV CONDA_BASE_PACKAGE $CONDA_BASE_PACKAGE
ENV CONDA_VERSION $CONDA_VERSION
RUN CONDA_BASE_PACKAGE_NAME=`echo $CONDA_BASE_PACKAGE | sed -r 's/\<./\U&/g'` \
    && OS=`uname -s` \
    && ARCH=`uname -m` \
    && CONDA_INSTALL_PACKAGE=$CONDA_BASE_PACKAGE_NAME$CONDA_PYTHON_VERSION-$CONDA_VERSION-$OS-$ARCH.sh \
    && echo "installing $CONDA_INSTALL_PACKAGE" \
    && if [ $CONDA_BASE_PACKAGE_NAME = 'Anaconda' ]; then \
         wget -q --show-progress --progress=bar:force:noscroll https://repo.anaconda.com/archive/$CONDA_INSTALL_PACKAGE -O /tmp/$CONDA_BASE_PACKAGE.sh ;\
       fi \
    && if [ $CONDA_BASE_PACKAGE_NAME = 'Miniconda' ]; then \
         wget -q --show-progress --progress=bar:force:noscroll https://repo.continuum.io/miniconda/$CONDA_INSTALL_PACKAGE -O /tmp/$CONDA_BASE_PACKAGE.sh ;\
       fi \
    && bash /tmp/$CONDA_BASE_PACKAGE.sh -b -p $HOME/$CONDA_BASE_PACKAGE$CONDA_PYTHON_VERSION \
    && rm /tmp/$CONDA_BASE_PACKAGE.sh

# labels
LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.name="edowson/vulkan-base/cudagl:$BUILD_VERSION-$CUDA_MAJOR_VERSION-xenial"
LABEL org.label-schema.description="Vulkan SDK $BUILD_VERSION with NVIDIA CUDA-$CUDA_VERSION CUDNN-$CUDNN_VERSION - Ubuntu-16.04."
LABEL org.label-schema.version=$BUILD_VERSION
LABEL org.label-schema.docker.cmd="xhost +local:root \
docker run -it \
  --runtime=nvidia \
  --device /dev/snd \
  -e DISPLAY \
  -e PULSE_SERVER=tcp:$HOST_IP:4713 \
  -e PULSE_COOKIE_DATA=`pax11publish -d | grep --color=never -Po '(?<=^Cookie: ).*'` \
  -e QT_GRAPHICSSYSTEM=native \
  -e QT_X11_NO_MITSHM=1 \
  -v /dev/shm:/dev/shm \
  -v /etc/localtime:/etc/localtime:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /var/run/dbus/system_bus_socket:/var/run/dbus/system_bus_socket:ro \
  -v ${XDG_RUNTIME_DIR}/pulse/native:/run/user/1000/pulse/native \
  -v ~/mount/backup:/backup \
  -v ~/mount/data:/data \
  -v ~/mount/project:/project \
  -v ~/mount/tool:/tool \
  --rm \
  --name vulkan-base-cudagl-$BUILD_VERSION-$CUDA_MAJOR_VERSION-xenial \
  edowson/vulkan-base/cudagl:$BUILD_VERSION-$CUDA_MAJOR_VERSION-xenial \
xhost -local:root"

# update .bashrc
RUN echo \
'CUDA="/usr/local/cuda-${CUDA_MAJOR_VERSION}"\n\
export PS1="${debian_chroot:+($debian_chroot)}\u:\W\$ "\n\
export VULKAN_SDK="/opt/vulkan/${VULKAN_SDK_VERSION}/x86_64"\n\
export VK_LAYER_PATH="${VULKAN_SDK}/etc/explicit_layer.d"\n\
export LD_LIBRARY_PATH="${VULKAN_SDK}/lib:${CUDA}/lib64:${CUDA}/extras/CUPTI/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"\n\
export PATH="${HOME}/bin:${VULKAN_SDK}/bin:${CUDA}/bin${PATH:+:${PATH}}"\n' \
    >> $HOME/.bashrc

COPY docker/ubuntu/xenial/vulkan-base/libm.so.6 /lib/x86_64-linux-gnu
COPY conda-env.txt /tmp

RUN $HOME/$CONDA_BASE_PACKAGE$CONDA_PYTHON_VERSION/bin/conda init bash

# update .condarc
RUN $HOME/$CONDA_BASE_PACKAGE$CONDA_PYTHON_VERSION/bin/conda config --set auto_activate_base false \
    && $HOME/$CONDA_BASE_PACKAGE$CONDA_PYTHON_VERSION/bin/conda config --add envs_dirs ~/.conda/env \
    && $HOME/$CONDA_BASE_PACKAGE$CONDA_PYTHON_VERSION/bin/conda config --add pkgs_dirs ~/.conda/pkgs \
    && $HOME/$CONDA_BASE_PACKAGE$CONDA_PYTHON_VERSION/bin/conda config --add channels anaconda \
    && $HOME/$CONDA_BASE_PACKAGE$CONDA_PYTHON_VERSION/bin/conda install --file /tmp/conda-env.txt

RUN pip install dlib insightface mxnet-cu92 facenet-pytorch --no-warn-script-location

RUN echo 'conda activate base \n' >> $HOME/.bashrc
