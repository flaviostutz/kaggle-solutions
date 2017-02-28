#!/bin/bash

echo ""
echo "================"
date
echo "Calling step3 script..."

jupyter nbconvert --to notebook --nbformat 1 --ExecutePreprocessor.timeout=-1 --execute datascience-snippets/ipython-notebooks/kaggle-lung-cancer-detection/step03-preparation-mask-center-resize-rotate.ipynb

echo "Done."
date
echo "================"
echo ""
