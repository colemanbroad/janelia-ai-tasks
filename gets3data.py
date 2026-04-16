import os
import fsspec, zarr
import dask.array as da # we import dask to help us manage parallel access to the big dataset
import ipdb
import numpy as np
from types import SimpleNamespace
from dask.diagnostics import ProgressBar
from dataclasses import dataclass

import torch

CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def loadN5(ds, subpath, idx, size_only=False):
    subpath = subpath.strip('/')
    cache_file = os.path.join(CACHE_DIR, f"{ds}_{subpath.replace('/', '_')}_{idx}.npy")
    if os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        return np.load(cache_file)

    path = f's3://janelia-cosem-datasets/{ds}/{ds}.n5'
    group = zarr.open(zarr.N5FSStore(path, anon=True))
    zdata = group
    for sub in subpath.split('/'):
        if not isinstance(zdata, zarr.hierarchy.Group):
            print(f"Reached a non-Group at '{sub}'. Remaining path can't be traversed.")
            return None
        if sub not in zdata:
            print(f"Key '{sub}' not found. Available keys: {list(zdata.keys())}")
            return None
        zdata = zdata[sub]
    if isinstance(zdata, zarr.hierarchy.Group):
        print(f"Path '{subpath}' is a Group, not an Array. Available keys: {list(zdata.keys())}")
        return None
    ddata = da.from_array(zdata, chunks=zdata.chunks)
    print(ddata)
    if size_only: return
    with ProgressBar():
        sli = idx if idx != 'all' else None
        result = ddata[sli].compute()
        np.save(cache_file, result)
        print(f"Saved to cache: {cache_file}")
    return result

def f3(w):
    # x = loadN5('jrc_fly-larva-1', 'em/tem-uint8/s3', 4816//2)
    # w.add_image(x, colormap='PiYG')
    x = loadN5('jrc_fly-larva-1', 'labels/s3', 4816//2)
    w.add_image(x, colormap='PiYG')
    # x = loadN5('jrc_mus-liver', 'em/fibsem-uint8/s3/', 1116//2, False)
    # w.add_image(x, colormap='PiYG')
    # x = loadN5('jrc_fly-larva-1', 'em/tem-uint8/s5', 300)
    # w.add_image(x, colormap='PiYG')
    # x = loadN5('jrc_jurkat-1', 'em/fibsem-uint16/s4', 100)
    # w.add_image(x, colormap='PiYG')


def f5():
    dinodir = "./../dinov3/"
    model = torch.hub.load(dinodir, 'dinov3_vits16', source='local', weights='dinoweights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
    model.eval()

    x = loadN5('jrc_mus-liver', 'em/fibsem-uint8/s3/', 1116//2, False)
    # Crop to patch-size-divisible dimensions, replicate to 3 channels, add batch dim
    H, W = (x.shape[0] // 16) * 16, (x.shape[1] // 16) * 16
    x_crop = x[:H, :W].astype(np.float32) / 255.0
    x_3ch = np.stack([x_crop] * 3)  # (3, H, W)
    x_tensor = torch.from_numpy(x_3ch).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        y = model(x_tensor)


    ipdb.set_trace()
    return y


