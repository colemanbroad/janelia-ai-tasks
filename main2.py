import os
import sys
import time
_t_start = time.time()
import argparse
from types import SimpleNamespace

import numpy as np
import torch
import zarr
import s3fs
import dask.array as da
from dask.diagnostics import ProgressBar
from sklearn.decomposition import PCA
from skimage.transform import resize
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

print(f"Imports done in {time.time() - _t_start:.1f}s", flush=True)

CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)
print("Ensured CACHE dir.")

if __name__ == "__main__":
    print("running main", flush=True)

# source .env && ssh -o StrictHostKeyChecking=no $HOST "cd $REMOTE_DIR && python main2.py"
