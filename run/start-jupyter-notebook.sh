#!/bin/bash

#KAGGLE CPU VERSION
docker run -d -m 13GB -v /Users/flaviostutz/Documents/development/flaviostutz/jupyter-notebooks:/root/workspace -w=/root/workspace -p 8888:8888 -p 6006:6006 --name jupyter -it kaggle/python jupyter notebook --no-browser --ip="*" --notebook-dir=/root/workspace

#TENSORFLOW NVIDIA GPU CONTAINER
#Install NVIDIA CUDA Drivers
#Install https://github.com/NVIDIA/nvidia-docker/
nvidia-docker run -d -m 13GB -v /Users/flaviostutz/Documents/development/flaviostutz/jupyter-notebooks:/root/workspace -w=/root/workspace -p 9999:8888 -p 7007:6006 --name jupyter-gpu -it tensorflow/tensorflow:0.12.1-gpu-py3 jupyter notebook --no-browser --ip="*" --notebook-dir=/root/workspace
