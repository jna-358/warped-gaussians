FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=Ampere

WORKDIR /content

ENV CC=/usr/bin/gcc-10
ENV CXX=/usr/bin/g++-10
ENV CUDAHOSTCXX=/usr/bin/g++-10

RUN apt-get update && \
    apt-get install -y git nano python3 python-is-python3  python3-pip && \
    apt-get install -y tmux gedit libglew-dev libassimp-dev libboost-all-dev && \
    apt-get install -y libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev && \
    apt-get install -y libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev && \ 
    apt-get install -y cmake ninja-build && \
    apt-get install -y python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    nvidia-cuda-toolkit \
    gcc-10 \
    g++-10 \
    nvidia-cuda-toolkit-gcc \
    && \
    rm -rf /var/lib/apt/lists/*

# Install 3DGS
WORKDIR /content
ARG CACHEBUST=1
RUN git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive

# Install python dependencies
RUN pip install plyfile torch torchvision torchaudio tqdm opencv-python numpy
RUN pip install matplotlib open3d scipy yaspin mmcv==1.6.0 argparse lpips pytorch-msssim
WORKDIR /content/gaussian-splatting
RUN pip install ./submodules/diff-gaussian-rasterization
RUN pip install ./submodules/simple-knn

# Clean up
RUN mv /content/gaussian-splatting/SIBR_viewers /content/viewers
RUN rm -rf /content/gaussian-splatting

RUN pip install debugpy
RUN pip install transformers

RUN pip install tensorboard
RUN pip install pandas

RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
RUN apt-get update && apt-get install -y --no-install-recommends fontconfig ttf-mscorefonts-installer && rm -rf /var/lib/apt/lists/*

# Default powerline10k theme, no plugins installed
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)"

ENV PROMPT_COMMAND='history -a'