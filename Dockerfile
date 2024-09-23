FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /root

# Install system dependencies
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
    liborocos-kdl-dev \
    libkdl-parser-dev \
    liburdfdom-dev \
    libnlopt-dev \
    libnlopt-cxx-dev \
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
  && make install \
  # hacky way to allow python ompl import
  && echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib" >> /root/.bashrc 

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && bash Miniconda3-latest-Linux-x86_64.sh -b \
  && rm -f Miniconda3-latest-Linux-x86_64.sh \
  && /root/miniconda3/bin/conda init \
  && /root/miniconda3/bin/conda create -n xmop_dev python=3.10.13 \
  && echo "conda activate xmop_dev" >> /root/.bashrc \
  && mkdir -p /root/miniconda3/envs/xmop_dev/etc/conda/activate.d \
  && echo "export PYTHONPATH=/root/xmop:$PYTHONPATH" >> /root/miniconda3/envs/xmop_dev/etc/conda/activate.d/env_vars.sh

WORKDIR /root