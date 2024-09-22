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
    unzip \
  && apt-get clean \
  && rm -rf /var/lib/apt

RUN python3 -m pip install -vU pygccxml pyplusplus \
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
  && /root/miniconda3/bin/conda create -n xmop_data_gen python=3.10.13 \
  && echo "conda activate xmop_data_gen" >> /root/.bashrc \
  # hacky way to allow python ompl import
  && echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib" >> /root/.bashrc 

# Install Dependencies
RUN /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install pybullet==3.2.6 pyquaternion==0.9.9 h5py==3.10.0 \
  && /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install torch --index-url https://download.pytorch.org/whl/cpu \
  && /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install geometrout==0.1.0.11 \
  && /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install urchin==0.0.27 \
  && /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install numpy==1.26.4

# Setup Codebase
RUN git clone https://github.com/prabinrath/xmop.git \
  && unzip xmop/dependencies/xacro_humble.zip \
  && /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install ./xacro_humble \
  && unzip xmop/dependencies/urdfpy-0.0.22.zip \
  && /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install ./urdfpy-0.0.22 \
  # networkx package installed by urdfpy has issues with python collections 
  && /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install networkx==3.1 \
  # install ompl python wheel to allow import
  && /root/miniconda3/bin/conda run -n xmop_data_gen pip3 install xmop/dependencies/ompl-1.6.0-cp310-cp310-manylinux_2_28_x86_64.whl \
  && mkdir -p xmop/log \
  && mkdir -p xmop/resources \
  && mkdir -p /root/miniconda3/envs/xmop_data_gen/etc/conda/activate.d \
  && echo "export PYTHONPATH=/root/xmop:$PYTHONPATH" >> /root/miniconda3/envs/xmop_data_gen/etc/conda/activate.d/env_vars.sh

WORKDIR /root/xmop