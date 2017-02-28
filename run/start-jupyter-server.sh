#!/bin/bash

nvidia-smi
if [ $? -eq 0 ]; then
    echo "USING TENSORFLOW WITH GPU SUPPORT"
    nvidia-docker run -d -v /root:/notebooks -v /root/input:/notebooks/input -v /root/output:/notebooks/output  -p 8888:8888 -p 6006:6006 --name jupyter-gpu -it tensorflow/tensorflow:1.0.0.2-gpu-py3

    nvidia-docker exec -d jupyter-gpu tensorboard --logdir=/workspace/output

else
    echo "USING TENSORFLOW WITHOUT GPU SUPPORT"
    docker run -d -v /root:/notebooks -v /root/input:/notebooks/input -v /root/output:/notebooks/output -p 8888:8888 -p 6006:6006 --name jupyter -it tensorflow/tensorflow:1.0.0

    docker exec -d jupyter tensorboard --logdir=/workspace/output

fi


#MACHINE PREPARATION FOR GPU NVIDA CUDA
# Install official NVIDIA driver package
#sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
#sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
#sudo apt-get update && sudo apt-get install -y --no-install-recommends cuda-drivers

# Install nvidia-docker and nvidia-docker-plugin
#wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.0/nvidia-docker_1.0.0-1_amd64.deb
#sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
