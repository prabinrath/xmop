#!/bin/bash

## install pip dependencies
pip install torch==2.2.0
pip install -r requirements.txt

## install urdfpy
unzip dependencies/urdfpy-0.0.22.zip -d dependencies/
pip install dependencies/urdfpy-0.0.22
pip install networkx==3.1

## install xacro
unzip dependencies/xacro_humble.zip -d dependencies/
pip install dependencies/xacro_humble

## install trackikpy
sudo apt-get install libeigen3-dev liborocos-kdl-dev libkdl-parser-dev liburdfdom-dev libnlopt-dev
pip install swig
pip install git+https://github.com/mjd3/tracikpy.git

## install ompl
pip install dependencies/ompl-1.6.0-cp310-cp310-manylinux_2_28_x86_64.whl