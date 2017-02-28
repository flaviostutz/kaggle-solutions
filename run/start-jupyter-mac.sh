#!/bin/bash

#KAGGLE CPU VERSION
docker run -d -m 13GB -v /Users/flaviostutz/Documents/development/flaviostutz/datascience-snippets/ipython-notebooks:/root/workspace -w=/root/workspace -p 8888:8888 -p 6006:6006 --name jupyter -it kaggle/python jupyter notebook --no-browser --ip="*" --notebook-dir=/root
