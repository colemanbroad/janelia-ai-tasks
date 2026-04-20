import os
import fsspec, zarr
import time
import dask.array as da # we import dask to help us manage parallel access to the big dataset
import ipdb
import numpy as np
import torch
from types import SimpleNamespace
from dask.diagnostics import ProgressBar
from dataclasses import dataclass
from sklearn.decomposition import PCA

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

    print(zdata)
    # ipdb.set_trace()
    ddata = da.from_array(zdata, chunks=zdata.chunks)
    print(ddata)
    if size_only: return
    with ProgressBar():
        sli = idx if idx != 'all' else None
        result = ddata[sli, 500:1000, 500:1000].compute()
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

def load_dino():
    dinodir = "./../dinov3/"
    model = torch.hub.load(dinodir, 'dinov3_vits16', source='local', weights='dinoweights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
    model.eval()
    return model

def run_dino(model, x_np):
    """Run DINO on a 2D grayscale numpy array. H and W must be divisible by 16."""
    x = x_np.astype(np.float32) # / 255.0
    mu = x_np.mean()
    std = x_np.std()
    x = (x - mu) / std 
    x = x * 0.229 + 0.485
    x = x_np.astype(np.float32) # / 255.0

    # ipdb.set_trace()

    x_3ch = np.stack([x] * 3)  # (3, H, W)
    x_tensor = torch.from_numpy(x_3ch).unsqueeze(0)  # (1, 3, H, W)

    t0 = time.time()
    with torch.no_grad():
        y = model.forward_features(x_tensor)
    dt = time.time() - t0
    print(f"Inference: {dt:.3f}s for input {x_np.shape}")
    return y

def run(w):
    res = f5()
    w.add_image(res[0])
    w.add_image(res[1])
    
def f5():
    model = load_dino()

    # Time on a small 16x16 crop
    # x = loadN5('jrc_mus-liver', 'em/fibsem-uint8/s3/', 2233//4, False)
    x = loadN5('jrc_mus-liver', 'em/fibsem-uint8/s2/', 2233//2, False)
    # x = loadN5('jrc_mus-liver', 'em/fibsem-uint8/s1/', 2233, False)
    # x = loadN5('jrc_mus-liver', 'em/fibsem-uint8/s0/', 2233, False)
    # print(x.shape)
    # x = x[500:1000, 500:1000]
    # return x
    # x = np.zeros((1591, 1593))
    # x = np.random.randn(1591, 1593)
    y_small = run_dino(model, x[:16, :16])
    print("Small crop output keys:", list(y_small.keys()))

    # Full slice — crop to patch-aligned dims
    H, W = (x.shape[0] // 16) * 16, (x.shape[1] // 16) * 16
    y_full = run_dino(model, x[:H, :W])

    # Per-patch embeddings: (num_patches, embed_dim)
    patch_tokens = y_full['x_norm_patchtokens'].squeeze(0)  # (N, 384)
    print(f"Patch tokens: {patch_tokens.shape}")

    # PCA on patch embeddings — take first 3 components as RGB
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(patch_tokens.numpy())  # (N, 3)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    # Normalize each component to [0, 1] for visualization
    for i in range(3):
        lo, hi = pca_features[:, i].min(), pca_features[:, i].max()
        pca_features[:, i] = (pca_features[:, i] - lo) / (hi - lo + 1e-8)

    # Reshape to patch grid
    pH, pW = H // 16, W // 16
    pca_grid = pca_features.reshape(pH, pW, 3)  # (pH, pW, 3)

    # Upscale: repeat each patch to 16x16 pixels
    pca_img = np.repeat(np.repeat(pca_grid, 16, axis=0), 16, axis=1)  # (H, W, 3)
    print(f"PCA image: {pca_img.shape}")

    return (x, pca_img)

# from transformers import pipeline
# from transformers.image_utils import load_image

def f6():
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image = load_image(url)

    feature_extractor = pipeline(
        model="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        task="image-feature-extraction",
    )
    features = feature_extractor(image)
    return features
    # ipdb.set_trace()
