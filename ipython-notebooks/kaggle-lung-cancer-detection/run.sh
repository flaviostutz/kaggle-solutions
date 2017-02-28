#!/bin/bash

docker exec -it datasciencetools_tensorflow-cpu_1 jupyter nbconvert --to notebook --nbformat 1 --ExecutePreprocessor.timeout=-1 --execute /notebooks/datascience-snippets/ipython-notebooks/kaggle-lung-cancer-detection/step03-preparation-mask-center-resize-rotate.ipynb

