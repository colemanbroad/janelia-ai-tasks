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
from sklearn.cluster import KMeans

CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def _slice_to_str(s):
    """Convert a slice/index tuple to a string for cache filenames."""
    if not isinstance(s, tuple):
        s = (s,)
    parts = []
    for x in s:
        if isinstance(x, slice):
            start = x.start if x.start is not None else ''
            stop = x.stop if x.stop is not None else ''
            step = x.step if x.step is not None else ''
            parts.append(f'{start}.{stop}.{step}')
        else:
            parts.append(str(x))
    return '_'.join(parts)

def load_remote(ds, subpath, slc, fmt='n5', size_only=False, refresh_cache=False):
    subpath = subpath.strip('/')
    slc_str = _slice_to_str(slc)
    cache_file = os.path.join(CACHE_DIR, f"{ds}_{fmt}_{subpath.replace('/', '_')}_{slc_str}.npy")
    if os.path.exists(cache_file) and not refresh_cache:
        print(f"Loading from cache: {cache_file}")
        return np.load(cache_file)

    if fmt == 'n5':
        path = f's3://janelia-cosem-datasets/{ds}/{ds}.n5'
        group = zarr.open(zarr.N5FSStore(path, anon=True))
    elif fmt == 'zarr':
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        path = f's3://janelia-cosem-datasets/{ds}/{ds}.zarr'
        group = zarr.open(zarr.storage.FSStore(path, fs=fs, mode='r'))
    else:
        assert False, f"Unknown format '{fmt}', use 'n5' or 'zarr'"
    zdata = group
    for sub in subpath.split('/'):
        assert isinstance(zdata, zarr.hierarchy.Group), f"Reached a non-Group at '{sub}'. Remaining path can't be traversed."
        assert sub in zdata, f"Key '{sub}' not found. Available keys: {list(zdata.keys())}"
        zdata = zdata[sub]
    assert not isinstance(zdata, zarr.hierarchy.Group), f"Path '{subpath}' is a Group, not an Array. Available keys: {list(zdata.keys())}"

    print(zdata)

    # Check bounds and clamp slices to array shape
    shape = zdata.shape
    slc_tuple = slc if isinstance(slc, tuple) else (slc,)
    clamped = []
    for dim, s in enumerate(slc_tuple):
        if isinstance(s, slice):
            lo = max(s.start if s.start is not None else 0, 0)
            hi = min(s.stop if s.stop is not None else shape[dim], shape[dim])
            assert lo < hi, f"Dim {dim}: slice [{s.start}:{s.stop}] is entirely out of bounds for size {shape[dim]}"
            if lo != s.start or hi != s.stop:
                print(f"Dim {dim}: clamped [{s.start}:{s.stop}] -> [{lo}:{hi}] (size {shape[dim]})")
            clamped.append(slice(lo, hi, s.step))
        else:
            assert 0 <= s < shape[dim], f"Dim {dim}: index {s} out of bounds for size {shape[dim]}"
            clamped.append(s)
    slc = tuple(clamped)

    # Remember original requested sizes for mirror-padding
    orig_slc = slc_tuple  # before clamping

    ddata = da.from_array(zdata, chunks=zdata.chunks)
    print(ddata)
    if size_only: return
    with ProgressBar():
        result = ddata[slc].compute()

    # Mirror-pad if clamping made the result smaller than requested
    pads = []
    needs_pad = False
    for dim, (orig, clamp) in enumerate(zip(orig_slc, slc)):
        if isinstance(orig, slice) and isinstance(clamp, slice):
            orig_lo = orig.start if orig.start is not None else 0
            orig_hi = orig.stop if orig.stop is not None else shape[dim]
            clamp_lo = clamp.start
            clamp_hi = clamp.stop
            pad_before = clamp_lo - orig_lo
            pad_after = orig_hi - clamp_hi
            if pad_before > 0 or pad_after > 0:
                needs_pad = True
            pads.append((max(pad_before, 0), max(pad_after, 0)))
        else:
            pads.append((0, 0))

    if needs_pad:
        # Only pad spatial dims (skip index dims that were removed)
        spatial_pads = [p for p, s in zip(pads, orig_slc) if isinstance(s, slice)]
        # result may have fewer dims than orig_slc if some were plain indices
        assert len(spatial_pads) == result.ndim, f"Pad mismatch: {len(spatial_pads)} pads vs {result.ndim} dims"
        result = np.pad(result, spatial_pads, mode='reflect')
        print(f"Mirror-padded: {pads}")

    np.save(cache_file, result)
    print(f"Saved to cache: {cache_file}")
    return result

def loadN5(ds, subpath, slc, **kwargs):
    return load_remote(ds, subpath, slc, fmt='n5', **kwargs)

def loadZarr(ds, subpath, slc, **kwargs):
    return load_remote(ds, subpath, slc, fmt='zarr', **kwargs)

def f3(w):
    c,b,a = 12057, 12301, 6229 ## copied from neuroglancer
    # x = loadZarr('jrc_mus-liver', 'recon-1/em/fibsem-uint8/s1', p2patch(a,b,c, s=1, const=0, hw=400), size_only=0, refresh_cache=0)

    c,b,a = 6332, 942, 5620
    # x = loadZarr('jrc_macrophage-2', 'recon-1/em/fibsem-uint8/s2', p2patch(a,b,c, s=2, const=0, hw=200), size_only=0, refresh_cache=0)

    c,b,a = 6417, 4150, 10157
    x = loadZarr('jrc_mus-kidney', 'recon-1/em/fibsem-uint8/s2', p2patch(a,b,c, s=2, const=0, hw=200), size_only=0, refresh_cache=0)
    mp = 165, 250 ## mito query location

    w.add_image(x)

def p2patch(a, b, c, s=0, const=0, hw=200):
    """Convert a neuroglancer center point to a tuple of slices.
    s: number of times to halve coords (for lower-res scales).
    const: which dimension (0,1,2) is held constant (the others become slices).
    hw: half-width of the slice window.
    """
    coords = [a, b, c]
    for _ in range(s):
        coords = [x // 2 for x in coords]
    print(coords)
    result = []
    for i, x in enumerate(coords):
        if i == const:
            result.append(x)
        else:
            result.append(slice(x - hw, x + hw))
    return tuple(result)

def load_dino():
    dinodir = "./../dinov3/"
    model = torch.hub.load(dinodir, 'dinov3_vits16', source='local', weights='dinoweights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
    model.eval()
    return model

def run_dino(model, x_np):
    """Run DINO on a 2D grayscale numpy array. H and W must be divisible by 16."""
    x = x_np.astype(np.float32)
    mu = x_np.mean()
    std = x_np.std()
    x = (x - mu) / std 
    x = x * 0.229 + 0.485
    x = x_np.astype(np.float32)

    x_3ch = np.stack([x] * 3)  # (3, H, W)
    x_tensor = torch.from_numpy(x_3ch).unsqueeze(0)  # (1, 3, H, W)

    t0 = time.time()
    with torch.no_grad():
        y = model.forward_features(x_tensor)
    dt = time.time() - t0
    print(f"Inference: {dt:.3f}s for input {x_np.shape}")
    return y

def test_estimate(stride=8):
  model = load_dino()
  model.patch_embed.proj.stride = (stride, stride)

  c,b,a = 12057, 12301, 6229
  img1 = loadZarr('jrc_mus-liver', 'recon-1/em/fibsem-uint8/s2', p2patch(a,b,c, s=2, const=0))

  c,b,a = 6417, 4150, 10157
  img2 = loadZarr('jrc_mus-kidney', 'recon-1/em/fibsem-uint8/s2', p2patch(a,b,c, s=2, const=0))

  for name, img in [('mus-liver', img1), ('mus-kidney', img2)]:
      print(f"=== {name} ({img.shape}) ===")
      est = estimate_inference_time(model, img)
      H, W = (img.shape[0] // 16) * 16, (img.shape[1] // 16) * 16
      t0 = time.time()
      run_dino(model, img[:H, :W].astype(np.float32))
      actual = time.time() - t0
      print(f"  estimated: {est:.1f}s, actual: {actual:.1f}s, ratio: {est/actual:.2f}")

def estimate_inference_time(model, img):
    """Estimate total inference time by running on small patches and extrapolating."""
    H, W = (img.shape[0] // 16) * 16, (img.shape[1] // 16) * 16
    n_full = (H // 16) * (W // 16)

    # Use two moderately sized crops to reduce warmup noise and capture quadratic scaling
    # n1 ~ 1/4 of patches, n2 ~ 1/2 of patches
    h1, w1 = max(H // 4, 16), max(W // 4, 16)
    h2, w2 = max(H // 2, 16), max(W // 2, 16)
    h1, w1 = (h1 // 16) * 16, (w1 // 16) * 16
    h2, w2 = (h2 // 16) * 16, (w2 // 16) * 16
    n1 = (h1 // 16) * (w1 // 16)
    n2 = (h2 // 16) * (w2 // 16)

    # Warmup with both sizes
    run_dino(model, img[:h1, :w1].astype(np.float32))
    run_dino(model, img[:h2, :w2].astype(np.float32))

    # Timed runs
    t0 = time.time()
    run_dino(model, img[:h1, :w1].astype(np.float32))
    t1 = time.time() - t0

    t0 = time.time()
    run_dino(model, img[:h2, :w2].astype(np.float32))
    t2 = time.time() - t0

    # Fit quadratic: time = a * n^2 + b * n, solve from two points
    # t1 = a*n1^2 + b*n1, t2 = a*n2^2 + b*n2
    det = n1**2 * n2 - n2**2 * n1
    if abs(det) > 1e-12:
        a = (t1 * n2 - t2 * n1) / det
        b = (t2 * n1**2 - t1 * n2**2) / det
        est = a * n_full**2 + b * n_full
    else:
        # Fallback to linear
        slope = (t2 - t1) / (n2 - n1)
        est = t1 + slope * (n_full - n1)

    print(f"  size1: {h1}x{w1} ({n1}p) = {t1:.3f}s, size2: {h2}x{w2} ({n2}p) = {t2:.3f}s")
    print(f"  full image ({H}x{W}, {n_full} patches): ~{est:.1f}s ({est/60:.1f}min)")
    return est

def runf5(w):
    res = f5()
    w.add_image(res[0])
    w.add_image(res[1])
    
def f5():
    model = load_dino()
    stride = 2
    model.patch_embed.proj.stride = (stride,stride)

    # Time on a small 16x16 crop
    # x = loadN5('jrc_mus-liver', 'em/fibsem-uint8/s3/', 2233//4, False)
    a, b = 16*30, 16*60
    ss = (2233//2, slice(a,b), slice(a,b))
    x = loadN5('jrc_mus-liver', 'em/fibsem-uint8/s2/', ss)
    # x = loadN5('jrc_mus-liver', 'em/fibsem-uint8/s1/', 2233, False)
    # x = loadN5('jrc_mus-liver', 'em/fibsem-uint8/s0/', 2233, False)
    # print(x.shape)
    # return x
    # x = np.zeros((1591, 1593))
    # x = np.random.randn(1591, 1593)

    y_small = run_dino(model, x[:16*2, :16*2])
    print("Small crop output keys:", list(y_small.keys()))
    ## TODO: make a dumb, approximate upper bound predition on inference time by extrapolating from
    # the times required to run inf on 1) a 16x16 patch and then 2) a 32x16 patch and linearly extrapolating.

    # ipdb.set_trace()

    # Full slice — crop to patch-aligned dims
    H, W = (x.shape[0] // 16) * 16, (x.shape[1] // 16) * 16
    y_full = run_dino(model, x[:H, :W])

    # Per-patch embeddings: (num_patches, embed_dim)
    patch_tokens = y_full['x_norm_patchtokens'].squeeze(0)  # (N, 384)
    print(f"Patch tokens: {patch_tokens.shape}")

    # PCA on patch embeddings — take first 3 components as RGB
    pca = PCA(n_components=3, whiten=False)
    pca_features = pca.fit_transform(patch_tokens.numpy())  # (N, 3)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    # Normalize each component to [0, 1] for visualization
    for i in range(3):
        lo, hi = pca_features[:, i].min(), pca_features[:, i].max()
        pca_features[:, i] = (pca_features[:, i] - lo) / (hi - lo + 1e-8)

    # With stride=4, patch grid is ((H-16)//4+1, (W-16)//4+1)
    # stride = 4
    patch_size = 16
    pH = (H - patch_size) // stride + 1
    pW = (W - patch_size) // stride + 1
    pca_grid = pca_features.reshape(pH, pW, 3)  # (pH, pW, 3)

    # Average overlapping patches: scatter each patch's value over its 16x16 region
    pca_img = np.zeros((H, W, 3), dtype=np.float64)
    counts = np.zeros((H, W, 1), dtype=np.float64)
    for i in range(pH):
        for j in range(pW):
            y0, x0 = i * stride, j * stride
            pca_img[y0:y0+patch_size, x0:x0+patch_size] += pca_grid[i, j]
            counts[y0:y0+patch_size, x0:x0+patch_size] += 1
    pca_img /= counts
    pca_img = pca_img.astype(np.float32)
    print(f"PCA image: {pca_img.shape}")

    return (x, pca_img)

def f7(w, k=8):
    """K-means on DINO patch embeddings, then tile patches grouped by cluster."""
    model = load_dino()

    a, b = 16*30, 16*60
    ss = (2233//2, slice(a,b), slice(a,b))
    x = loadN5('jrc_mus-liver', 'em/fibsem-uint8/s2/', ss)
    H, W = (x.shape[0] // 16) * 16, (x.shape[1] // 16) * 16
    x_crop = x[:H, :W]

    y = run_dino(model, x_crop)
    patch_tokens = y['x_norm_patchtokens'].squeeze(0).numpy()  # (N, 384)
    pH, pW = H // 16, W // 16

    # K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(patch_tokens)  # (N,)
    label_grid = labels.reshape(pH, pW)

    # Extract 16x16 patches from the original image
    patches = x_crop.reshape(pH, 16, pW, 16).transpose(0, 2, 1, 3)  # (pH, pW, 16, 16)

    # For each cluster, collect its patches and tile them into a grid
    cluster_images = []
    for c in range(k):
        mask = labels == c
        cluster_patches = patches.reshape(-1, 16, 16)[mask]  # (n_c, 16, 16)
        n_c = len(cluster_patches)
        if n_c == 0:
            continue
        cols = int(np.ceil(np.sqrt(n_c)))
        rows = int(np.ceil(n_c / cols))
        # Pad to fill the grid
        pad_count = rows * cols - n_c
        if pad_count > 0:
            padding = np.zeros((pad_count, 16, 16), dtype=cluster_patches.dtype)
            cluster_patches = np.concatenate([cluster_patches, padding])
        tile = cluster_patches.reshape(rows, cols, 16, 16).transpose(0, 2, 1, 3).reshape(rows * 16, cols * 16)
        cluster_images.append(tile)

    # Show label map and cluster tiles
    label_img = np.repeat(np.repeat(label_grid, 16, axis=0), 16, axis=1)
    w.add_image(x_crop, name='original')
    w.add_image(label_img, name='cluster_labels', colormap='turbo')
    for i, img in enumerate(cluster_images):
        w.add_image(img, name=f'cluster_{i}')

def overlap_average(feature_grid, H, W, patch_size, stride):
    """Average overlapping patches back into an image.
    feature_grid: (pH, pW, C), returns (H, W, C)."""
    C = feature_grid.shape[2]
    out = np.zeros((H, W, C), dtype=np.float64)
    counts = np.zeros((H, W, 1), dtype=np.float64)
    pH, pW = feature_grid.shape[:2]
    for i in range(pH):
        for j in range(pW):
            y0, x0 = i * stride, j * stride
            out[y0:y0+patch_size, x0:x0+patch_size] += feature_grid[i, j]
            counts[y0:y0+patch_size, x0:x0+patch_size] += 1
    out /= counts
    return out.astype(np.float32)

def lbp_patch_histograms(img, radius=1, n_points=8, patch_size=16, stride=2):
    """Compute LBP histogram features for overlapping patches.
    Returns (pH, pW, n_bins) array of normalized histograms."""
    from skimage.feature import local_binary_pattern
    H, W = img.shape[:2]
    pH = (H - patch_size) // stride + 1
    pW = (W - patch_size) // stride + 1
    lbp_img = local_binary_pattern(img, n_points, radius, method='uniform')
    n_bins = n_points + 2
    features = np.zeros((pH, pW, n_bins), dtype=np.float32)
    for i in range(pH):
        for j in range(pW):
            y0, x0 = i * stride, j * stride
            patch = lbp_img[y0:y0+patch_size, x0:x0+patch_size]
            hist, _ = np.histogram(patch, bins=n_bins, range=(0, n_bins), density=True)
            features[i, j] = hist
    return features


def f8test(w):
  # Mouse liver
  c,b,a = 12057, 12301, 6229
  img1 = loadZarr('jrc_mus-liver', 'recon-1/em/fibsem-uint8/s2', p2patch(a,b,c, s=2, const=0))
  f8(w, img1, stride=8, name='mus-liver')

  # Another dataset
  c,b,a = 6417, 4150, 10157
  img2 = loadZarr('jrc_mus-kidney', 'recon-1/em/fibsem-uint8/s2', p2patch(a,b,c, s=2, const=0))
  f8(w, img2, stride=8, name='mus-kidney')

def f8(w, img, stride=2, name=''):
    """Compare DINO PCA vs LBP histogram PCA on overlapping patches.
    img: 2D numpy array (grayscale)."""
    patch_size = 16

    H, W = (img.shape[0] // 16) * 16, (img.shape[1] // 16) * 16
    x_crop = img[:H, :W].astype(np.float32)

    # --- LBP ---
    H, W = x_crop.shape[:2]
    pH = (H - patch_size) // stride + 1
    pW = (W - patch_size) // stride + 1
    # lbp_features = lbp_patch_histograms(x_crop, stride=stride)
    # pH, pW, n_bins = lbp_features.shape

    # pca_lbp = PCA(n_components=3)
    # lbp_pca = pca_lbp.fit_transform(lbp_features.reshape(-1, n_bins))
    # for i in range(3):
    #     lo, hi = lbp_pca[:, i].min(), lbp_pca[:, i].max()
    #     lbp_pca[:, i] = (lbp_pca[:, i] - lo) / (hi - lo + 1e-8)
    # lbp_pca_grid = lbp_pca.reshape(pH, pW, 3)
    # lbp_pca_img = overlap_average(lbp_pca_grid, H, W, patch_size, stride)

    # --- DINO ---
    model = load_dino()
    model.patch_embed.proj.stride = (stride, stride)
    y_full = run_dino(model, x_crop)
    patch_tokens = y_full['x_norm_patchtokens'].squeeze(0).numpy()

    pca_dino = PCA(n_components=3)
    dino_pca = pca_dino.fit_transform(patch_tokens)
    for i in range(3):
        lo, hi = dino_pca[:, i].min(), dino_pca[:, i].max()
        dino_pca[:, i] = (dino_pca[:, i] - lo) / (hi - lo + 1e-8)
    dino_pca_grid = dino_pca.reshape(pH, pW, 3)
    dino_pca_img = overlap_average(dino_pca_grid, H, W, patch_size, stride)

    pfx = f'{name} ' if name else ''
    w.add_image(x_crop, name=f'{pfx}original')
    # w.add_image(lbp_pca_img, name=f'{pfx}LBP_PCA', rgb=True)
    w.add_image(dino_pca_img, name=f'{pfx}DINO_PCA', rgb=True)

def f9(w, stride=4, patch_size=16):
    """Test LBP with different (radius, n_points) combos side by side."""
    a, b = 16*30, 16*60
    ss = (2233*2, slice(a*4,b*4), slice(a*4,b*4))
    x = loadN5('jrc_mus-liver', 'em/fibsem-uint8/s0/', ss)
    H, W = (x.shape[0] // 16) * 16, (x.shape[1] // 16) * 16
    x_crop = x[:H, :W].astype(np.float32)

    pH = (H - patch_size) // stride + 1
    pW = (W - patch_size) // stride + 1

    configs = [
        (1, 8),
        (2, 16),
        (3, 24),
        (4, 32),
        (1, 16),
        (2, 8),
        (3, 8),
        (5, 40),
    ]

    w.add_image(x_crop, name='original')

    for radius, n_points in configs:
        t0 = time.time()
        lbp_features = lbp_patch_histograms(x_crop, radius=radius, n_points=n_points, patch_size=patch_size, stride=stride)
        pH, pW, n_bins = lbp_features.shape

        pca = PCA(n_components=3)
        pca_out = pca.fit_transform(lbp_features.reshape(-1, n_bins))
        for c in range(3):
            lo, hi = pca_out[:, c].min(), pca_out[:, c].max()
            pca_out[:, c] = (pca_out[:, c] - lo) / (hi - lo + 1e-8)

        pca_grid = pca_out.reshape(pH, pW, 3)
        pca_img = overlap_average(pca_grid, H, W, patch_size, stride)

        dt = time.time() - t0
        name = f'LBP_r{radius}_p{n_points} ({dt:.1f}s)'
        print(f"{name}  variance: {pca.explained_variance_ratio_}")
        w.add_image(pca_img, name=name, rgb=True)


def task1(n_samples=20, crop_size=1024):
    """Download random 2D crops from liver and kidney datasets at s0.
    All crops are exactly crop_size x crop_size and fully within the volume."""
    np.random.seed(42)

    datasets = {
        'liver': {
            'ds': 'jrc_mus-liver',
            'subpath': 'recon-1/em/fibsem-uint8/s0',
        },
        'kidney': {
            'ds': 'jrc_mus-kidney',
            'subpath': 'recon-1/em/fibsem-uint8/s0',
        },
    }

    for dname, info in datasets.items():
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        path = f's3://janelia-cosem-datasets/{info["ds"]}/{info["ds"]}.zarr'
        group = zarr.open(zarr.storage.FSStore(path, fs=fs, mode='r'))
        zdata = group
        for sub in info['subpath'].split('/'):
            zdata = zdata[sub]
        shape = zdata.shape  # (z, y, x)
        print(f"{dname}: volume shape = {shape}")

        zs = np.random.randint(0, shape[0], size=n_samples)
        ys = np.random.randint(0, shape[1] - crop_size, size=n_samples)
        xs = np.random.randint(0, shape[2] - crop_size, size=n_samples)

        out_dir = os.path.join(CACHE_DIR, f'task1_{dname}')
        os.makedirs(out_dir, exist_ok=True)

        for i, (z, y, x) in enumerate(zip(zs, ys, xs)):
            out_file = os.path.join(out_dir, f'{i:02d}_z{z}_y{y}_x{x}.npy')
            if os.path.exists(out_file):
                print(f"  [{i}] already exists: {out_file}")
                continue
            slc = (int(z), slice(int(y), int(y + crop_size)), slice(int(x), int(x + crop_size)))
            img = load_remote(info['ds'], info['subpath'], slc, fmt='zarr')
            np.save(out_file, img)
            print(f"  [{i}] saved {img.shape} -> {out_file}")

        print(f"{dname}: {n_samples} crops saved to {out_dir}")

def load_datasets():
    """Load all task1 crops into a dict keyed by dataset name.
    Returns {'liver': {'images': [np arrays], 'coords': [(z,y,x), ...]},
             'kidney': {...}}"""
    result = {}
    for dname in ['liver', 'kidney']:
        out_dir = os.path.join(CACHE_DIR, f'task1_{dname}')
        assert os.path.isdir(out_dir), f"No data found at {out_dir}. Run task1() first."
        files = sorted([f for f in os.listdir(out_dir) if f.endswith('.npy')])
        images = []
        coords = []
        for f in files:
            # Parse coords from filename: 00_z123_y456_x789.npy
            parts = f.replace('.npy', '').split('_')
            z = int(parts[1][1:])
            y = int(parts[2][1:])
            x = int(parts[3][1:])
            images.append(np.load(os.path.join(out_dir, f)))
            coords.append((z, y, x))
        result[dname] = {'images': images, 'coords': coords}
        print(f"{dname}: loaded {len(images)} images, shape={images[0].shape}")
    return result

def show_datasets(w):
    data = load_datasets()
    for dname in ['liver', 'kidney']:
        stack = np.stack(data[dname]['images'])  # (N, H, W)
        w.add_image(stack, name=dname)

def task2(w, stride=16, downsample_factor=1, n_images=3):
    """Run DINO on images, per-image mean subtraction, per-image PCA, scrollable in napari.
    downsample_factor: locally downscale s0 images by this factor before running DINO.
    n_images: how many images per dataset to use."""
    from skimage.transform import resize
    patch_size = 16
    model = load_dino()
    model.patch_embed.proj.stride = (stride, stride)

    data = load_datasets()

    for dname in ['liver', 'kidney']:
        images = data[dname]['images'][:n_images]

        # Downscale and crop to patch-aligned dims
        crops = []
        for img in images:
            img = img.astype(np.float32)
            if downsample_factor > 1:
                new_h, new_w = img.shape[0] // downsample_factor, img.shape[1] // downsample_factor
                img = resize(img, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(np.float32)
            H, W = (img.shape[0] // 16) * 16, (img.shape[1] // 16) * 16
            crops.append(img[:H, :W])

        H, W = crops[0].shape
        pH = (H - patch_size) // stride + 1
        pW = (W - patch_size) // stride + 1

        # Run DINO, per-image mean subtraction, per-image PCA
        raw_stack = []
        pca_stack = []
        for i, x_crop in enumerate(crops):
            print(f"{dname} [{i}] inference...")
            y_full = run_dino(model, x_crop)
            tokens = y_full['x_norm_patchtokens'].squeeze(0).numpy()

            # Per-image mean subtraction
            # tokens = tokens - tokens.mean(axis=0, keepdims=True)
            print(tokens.mean(axis=0, keepdims=True))
            ipdb.set_trace()
            continue

            # Per-image PCA
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(tokens)
            print(f"  PCA variance: {pca.explained_variance_ratio_}")

            for c in range(3):
                lo, hi = pca_features[:, c].min(), pca_features[:, c].max()
                pca_features[:, c] = (pca_features[:, c] - lo) / (hi - lo + 1e-8)

            pca_grid = pca_features.reshape(pH, pW, 3)
            pca_tensor = torch.from_numpy(pca_grid).permute(2, 0, 1).unsqueeze(0)
            pca_img = torch.nn.functional.interpolate(pca_tensor, size=(H, W), mode='nearest', align_corners=None)
            pca_img = pca_img.squeeze(0).permute(1, 2, 0).numpy()

            raw_stack.append(x_crop)
            pca_stack.append(pca_img)

        raw_stack = np.stack(raw_stack)
        pca_stack = np.stack(pca_stack)

        label = f'{dname} {downsample_factor}x'
        w.add_image(raw_stack, name=f'{label} raw')
        w.add_image(pca_stack, name=f'{label} PCA', rgb=True)

def task3(w, stride=2):
    from skimage.feature import peak_local_max
    patch_size = 16

    model = load_dino()
    model.patch_embed.proj.stride = (stride, stride)

    # Load liver and kidney images
    # c,b,a = 12057, 12301, 6229
    c,b,a = 3754, 2619, 3515
    img_liver = loadZarr('jrc_mus-liver', 'recon-1/em/fibsem-uint8/s2', p2patch(a,b,c, s=2, const=0))
    c,b,a = 6417, 4150, 10157
    img_kidney = loadZarr('jrc_mus-kidney', 'recon-1/em/fibsem-uint8/s2', p2patch(a,b,c, s=2, const=0))

    # Mito query points (y, x) in each image
    liver_mito = (44, 163)
    kidney_mito = (165, 250)

    datasets = {
        'liver':  (img_liver,  liver_mito),
        'kidney': (img_kidney, kidney_mito),
    }

    # Get stride=2 embeddings for each image: (pH, pW, 384)
    embeddings = {}
    crops = {}
    for name, (img, _) in datasets.items():
        H, W = (img.shape[0] // 16) * 16, (img.shape[1] // 16) * 16
        x_crop = img[:H, :W].astype(np.float32)
        y_full = run_dino(model, x_crop)
        tokens = y_full['x_norm_patchtokens'].squeeze(0)  # (N, 384)
        pH = (H - patch_size) // stride + 1
        pW = (W - patch_size) // stride + 1
        embeddings[name] = (tokens, pH, pW, H, W)
        crops[name] = x_crop

    # Get query embedding at mito centerpoint for each dataset
    queries = {}
    for name, (img, mito_yx) in datasets.items():
        tokens, pH, pW, H, W = embeddings[name]
        qy, qx = mito_yx
        qi = min(qy // stride, pH - 1)
        qj = min(qx // stride, pW - 1)
        queries[name] = torch.nn.functional.normalize(tokens[qi * pW + qj].unsqueeze(0), dim=-1)

    # Add raw images
    # w.add_image(crops['liver'], name='liver original')
    # w.add_image(crops['kidney'], name='kidney original')

    # Compute all 4 combinations: query from {liver,kidney} x target {liver,kidney}
    for q_name in ['liver', 'kidney']:
        query_emb = queries[q_name]
        q_mito = datasets[q_name][1]
        w.add_points(np.array([list(q_mito)]), name=f'{q_name} query', size=10, face_color='red')

        for t_name in ['liver', 'kidney']:
            tokens, pH, pW, H, W = embeddings[t_name]
            patch_normed = torch.nn.functional.normalize(tokens, dim=-1)
            cos_sim = (patch_normed @ query_emb.T).squeeze(-1).numpy()

            sim_grid = cos_sim.reshape(pH, pW, 1)
            sim_img = overlap_average(sim_grid, H, W, patch_size, stride).squeeze(-1)

            peaks = peak_local_max(sim_img, min_distance=10, threshold_rel=0.3)
            label = f'query={q_name} target={t_name}'
            print(f"{label}: {len(peaks)} detections")


            # w.add_image(crops[t_name], name=t_name + ' original')
            # w.add_image(sim_img, name=f'{label} sim', colormap='inferno')
            w.add_points(peaks, name=f'{label} detections', size=8, face_color='green')

def mito_retrieval(w, img, query_yx, stride=2, min_distance=10, threshold_rel=0.3, name=''):
    """Cosine similarity retrieval from a query mito point.
    img: 2D grayscale numpy array.
    query_yx: (y, x) pixel location of query mitochondrion."""
    from skimage.feature import peak_local_max
    patch_size = 16

    model = load_dino()
    model.patch_embed.proj.stride = (stride, stride)

    H, W = (img.shape[0] // 16) * 16, (img.shape[1] // 16) * 16
    x_crop = img[:H, :W].astype(np.float32)

    y_full = run_dino(model, x_crop)
    patch_tokens = y_full['x_norm_patchtokens'].squeeze(0)  # (N, 384)
    patch_normed = torch.nn.functional.normalize(patch_tokens, dim=-1)

    pH = (H - patch_size) // stride + 1
    pW = (W - patch_size) // stride + 1

    # Map query pixel to patch index
    qy, qx = query_yx
    qi = min(qy // stride, pH - 1)
    qj = min(qx // stride, pW - 1)
    query_emb = torch.nn.functional.normalize(patch_tokens[qi * pW + qj].unsqueeze(0), dim=-1)

    # Cosine similarity
    cos_sim = (patch_normed @ query_emb.T).squeeze(-1).numpy()

    sim_grid = cos_sim.reshape(pH, pW, 1)
    sim_img = overlap_average(sim_grid, H, W, patch_size, stride).squeeze(-1)

    # Detect peaks
    peaks = peak_local_max(sim_img, min_distance=min_distance, threshold_rel=threshold_rel)
    print(f"Found {len(peaks)} local maxima")

    pfx = f'{name} ' if name else ''
    w.add_image(x_crop, name=f'{pfx}original')
    w.add_image(sim_img, name=f'{pfx}cosine_sim', colormap='inferno')
    w.add_points(np.array([list(query_yx)]), name=f'{pfx}query', size=10, face_color='red')
    w.add_points(peaks, name=f'{pfx}detections', size=8, face_color='green')

    return sim_img, peaks

def f10test(w):
    # Mouse liver
    c,b,a = 12057, 12301, 6229
    img1 = loadZarr('jrc_mus-liver', 'recon-1/em/fibsem-uint8/s2', p2patch(a,b,c, s=2, const=0))
    mito_retrieval(w, img1, query_yx=(44, 163), name='liver')

    # Mouse kidney
    # c,b,a = 6417, 4150, 10157
    # img2 = loadZarr('jrc_mus-kidney', 'recon-1/em/fibsem-uint8/s2', p2patch(a,b,c, s=2, const=0))
    # mito_retrieval(w, img2, query_yx=(165, 250), name='kidney')

def f11(w, sigma_range=(1, 3, 5, 9, 15)):
    """Blur + threshold segmentation sweep on mus-kidney image."""
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu

    c, b, a = 6417, 4150, 10157
    img = loadZarr('jrc_mus-kidney', 'recon-1/em/fibsem-uint8/s2', p2patch(a, b, c, s=2, const=0, hw=200))
    img = img.astype(np.float32)

    w.add_image(img, name='original')

    from scipy.ndimage import label as ndlabel

    for sigma in sigma_range:
        blurred = gaussian_filter(img, sigma=sigma)
        thresh = threshold_otsu(blurred)
        mask = blurred < thresh  # mitos are dark in EM
        labels, n_objects = ndlabel(mask)
        w.add_image(blurred, name=f'blur_s{sigma}')
        w.add_labels(labels, name=f'seg_s{sigma}')
        print(f"sigma={sigma}: threshold={thresh:.1f}, mito_frac={mask.mean():.3f}, n_objects={n_objects}")


