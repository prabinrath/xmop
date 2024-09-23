#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <option>"
    exit 1
fi

# Assign the argument to a variable
OPTION=$1

# Run wget commands based on the value of the argument
if [ "$OPTION" == "xcod" ]; then
    echo "Downloading XCoD dataset"
    mkdir -p resources/datasets/coll_dataset
    wget -O resources/datasets/coll_dataset/xcod_train.zip --show-progress https://huggingface.co/datasets/prabinrath/xmop/resolve/main/xcod_train.zip
    unzip resources/datasets/coll_dataset/xcod_train.zip -d resources/datasets/coll_dataset
    rm resources/datasets/coll_dataset/xcod_train.zip

elif [ "$OPTION" == "xmop" ]; then
    echo "Downloading XMoP dataset"
    mkdir -p resources/datasets/traj_dataset
    wget -O resources/datasets/traj_dataset/xmop_train.zip --show-progress https://huggingface.co/datasets/prabinrath/xmop/resolve/main/xmop_train.zip
    unzip resources/datasets/traj_dataset/xmop_train.zip -d resources/datasets/traj_dataset
    rm resources/datasets/traj_dataset/xmop_train.zip

elif [ "$OPTION" == "mpinet" ]; then
    echo "Downloading MpiNet dataset"
    mkdir -p resources/datasets/mpinet_dataset
    wget -O resources/datasets/mpinet_dataset/mpinet_train.zip --show-progress https://huggingface.co/datasets/prabinrath/xmop/resolve/main/mpinet_train.zip
    unzip resources/datasets/mpinet_dataset/mpinet_train.zip -d resources/datasets/mpinet_dataset
    rm resources/datasets/mpinet_dataset/mpinet_train.zip

elif [ "$OPTION" == "benchmark" ]; then
    echo "Downloading Benchmark problems"
    mkdir -p resources/
    wget --show-progress https://huggingface.co/datasets/prabinrath/xmop/resolve/main/benchmark_problems.zip -P resources/
    unzip resources/benchmark_problems.zip -d resources/
    rm resources/benchmark_problems.zip

else
    echo "Invalid option: $OPTION"
    echo "Valid options are: xcod, xmop, mpinet"
    exit 1
fi

echo "Download complete!"