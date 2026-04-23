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

    ddata = da.from_array(zdata, chunks=zdata.chunks)
    print(ddata)
    if size_only: return
    with ProgressBar():
        result = ddata[slc].compute()
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

def run(w):
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

def f10(w, query_yx=(107, 317), bg_yx=(232, 230), stride=2):
    """Task 2.3.1: Embedding-based retrieval using a query mito point.
    Compute similarity to query mito minus similarity to background point."""
    patch_size = 16
    model = load_dino()
    model.patch_embed.proj.stride = (stride, stride)

    a, b = 16*30, 16*60
    ss = (2233//2, slice(a,b), slice(a,b))
    x = loadN5('jrc_mus-liver', 'em/fibsem-uint8/s2/', ss)
    H, W = (x.shape[0] // 16) * 16, (x.shape[1] // 16) * 16
    x_crop = x[:H, :W]

    y_full = run_dino(model, x_crop.astype(np.float32))
    patch_tokens = y_full['x_norm_patchtokens'].squeeze(0)  # (N, 384)
    patch_normed = torch.nn.functional.normalize(patch_tokens, dim=-1)

    pH = (H - patch_size) // stride + 1
    pW = (W - patch_size) // stride + 1

    def pixel_to_patch(yx):
        y, x = yx
        return min(y // stride, pH - 1), min(x // stride, pW - 1)

    # Query mito embedding
    qi, qj = pixel_to_patch(query_yx)
    query_emb = torch.nn.functional.normalize(patch_tokens[qi * pW + qj].unsqueeze(0), dim=-1)
    sim_mito = (patch_normed @ query_emb.T).squeeze(-1).numpy()

    # Background embedding
    bi, bj = pixel_to_patch(bg_yx)
    bg_emb = torch.nn.functional.normalize(patch_tokens[bi * pW + bj].unsqueeze(0), dim=-1)
    sim_bg = (patch_normed @ bg_emb.T).squeeze(-1).numpy()

    # Mito score = similarity to mito - similarity to background
    score = sim_mito - sim_bg

    # Visualize
    score_grid = score.reshape(pH, pW, 1)
    score_img = overlap_average(score_grid, H, W, patch_size, stride).squeeze(-1)

    # Find local maxima on the combined score
    from skimage.feature import peak_local_max
    peaks = peak_local_max(score_img, min_distance=10, threshold_rel=0.3)
    print(f"Found {len(peaks)} local maxima")

    w.add_image(x_crop, name='original')
    w.add_image(score_img, name='mito_score', colormap='inferno')
    w.add_points(np.array([list(query_yx)]), name='query_mito', size=10, face_color='red')
    w.add_points(np.array([list(bg_yx)]), name='query_bg', size=10, face_color='blue')
    w.add_points(peaks, name='similar_peaks', size=8, face_color='green')

    return score_img

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
