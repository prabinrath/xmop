FROM ubuntu:jammy

WORKDIR /root

# Install OMPL
ENV DEBIAN_FRONTEND="noninteractive" TZ="America/Phoenix"
RUN apt update && apt install -y --no-install-recommends software-properties-common 
RUN apt install -y --no-install-recommends \
    g++ \
    cmake \
    pkg-config \
    libboost-serialization-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libboost-test-dev \
    libboost-python-dev \
    libboost-numpy-dev \
    libeigen3-dev \
    libode-dev \
    wget \
    libyaml-cpp-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    castxml \
    pypy3 \
  && apt-get clean \
  && rm -rf /var/lib/apt

RUN python3 -m pip install -vU https://github.com/CastXML/pygccxml/archive/develop.zip pyplusplus numpy \
  && git clone --recurse-submodules https://github.com/ompl/ompl.git \
  && export CXX=g++ \
  && export MAKEFLAGS="-j `nproc`" \
  && mkdir -p ompl/build/Release \
  && cd ompl/build/Release \
  && cmake ../.. -DPYTHON_EXEC=/usr/bin/python3 \
  && make update_bindings \
  && make \
  && make install

# Install CONDA
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && bash Miniconda3-latest-Linux-x86_64.sh -b \
  && rm -f Miniconda3-latest-Linux-x86_64.sh \
  && /root/miniconda3/bin/conda init \
  && /root/miniconda3/bin/conda create -n xmop_data_gen python=3.10.6 \
  && echo "conda activate xmop_data_gen" >> /root/.bashrc 

# Install Task Dependencies
RUN /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install pybullet pyquaternion h5py \
  && /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install torch --index-url https://download.pytorch.org/whl/cpu \
  && /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install geometrout==0.1.0.11 \
  && /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install urchin

RUN apt update && apt install -y --no-install-recommends unzip

# Setup Codebase and Dataset
RUN git clone https://prabinrath/xmop.git \
  && mkdir -p xmop/traj_dataset \
  && mkdir -p xmop/log \
  && unzip xmop/dependencies/xacro_humble.zip \
  && /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install ./xacro_humble \
  && unzip xmop/dependencies/urdfpy-0.0.22.zip \
  && /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install ./urdfpy-0.0.22 \
  # networkx package installed by urdfpy has issues with python collections 
  && /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install networkx==3.1 \
  && mkdir -p /root/miniconda3/envs/xmop_data_gen/etc/conda/activate.d \
  && echo "export PYTHONPATH=/root/xmop:$PYTHONPATH" >> /root/miniconda3/envs/xmop_data_gen/etc/conda/activate.d/env_vars.sh

WORKDIR /root/xmop