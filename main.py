import os
import zarr
import time
import matplotlib
matplotlib.use('Agg')  # headless-safe backend, must be before pyplot import
import dask.array as da # we import dask to help us manage parallel access to the big dataset
import numpy as np
import torch
from dask.diagnostics import ProgressBar
from sklearn.decomposition import PCA

CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

dataset_names = [
    'jrc_hela-3',
    'jrc_fly-larva-1',
    'jrc_jurkat-1',
    'jrc_hela-2',
    'jrc_macrophage-2',
    'jrc_mus-liver',
    'jrc_hela-1',
    'jrc_ccl81-covid-1',
    'jrc_choroid-plexus-2',
    'jrc_cos7-11',
    'jrc_ctl-id8-1',
    'jrc_ctl-id8-2',
    'jrc_ctl-id8-3',
    'jrc_ctl-id8-4',
    'jrc_ctl-id8-5',
    'jrc_dauer-larva',
    'jrc_fly-acc-calyx-1',
    'jrc_fly-fsb-1',
    'jrc_fly-mb-z0419-20',
    'jrc_hela-21',
    'jrc_hela-22',
    'jrc_hela-4',
    'jrc_hela-bfa',
    'jrc_hela-h89-1',
    'jrc_hela-h89-2',
    'jrc_mus-kidney',
    'jrc_mus-pancreas-1',
    'jrc_mus-pancreas-2',
    'jrc_mus-pancreas-3',
    'jrc_mus-sc-zp104a',
    'jrc_mus-sc-zp105a',
    'jrc_sum159-1',
]

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

DINO_MODELS = {
    # ViT LVD-1689M
    'vits16': {
        'hub_name': 'dinov3_vits16',
        'weights': 'dinoweights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
    },
    'vits16plus': {
        'hub_name': 'dinov3_vits16plus',
        'weights': 'dinoweights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
    },
    'vitb16': {
        'hub_name': 'dinov3_vitb16',
        'weights': 'dinoweights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    },
    'vitl16_lvd': {
        'hub_name': 'dinov3_vitl16',
        'weights': 'dinoweights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
    },
    'vith16plus': {
        'hub_name': 'dinov3_vith16plus',
        'weights': 'dinoweights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
    },
    'vit7b16': {
        'hub_name': 'dinov3_vit7b16',
        'weights': 'dinoweights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth',
    },
    # ViT SAT-493M
    'vitl16': {
        'hub_name': 'dinov3_vitl16',
        'weights': 'dinoweights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
    },
    'vit7b16_sat': {
        'hub_name': 'dinov3_vit7b16',
        'weights': 'dinoweights/dinov3_vit7b16_pretrain_sat493m-a6675841.pth',
    },
    # ConvNeXt LVD-1689M
    'convnext_tiny': {
        'hub_name': 'dinov3_convnext_tiny',
        'weights': 'dinoweights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',
    },
    'convnext_small': {
        'hub_name': 'dinov3_convnext_small',
        'weights': 'dinoweights/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth',
    },
    'convnext_base': {
        'hub_name': 'dinov3_convnext_base',
        'weights': 'dinoweights/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth',
    },
    'convnext_large': {
        'hub_name': 'dinov3_convnext_large',
        'weights': 'dinoweights/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth',
    },
}

def detect_gpu():
    """Return GPU name and memory in GB, or None if no GPU."""
    if not torch.cuda.is_available():
        return None, 0
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return name, mem_gb

def load_dino(model_name=None):
    if model_name is None:
        model_name = os.environ.get('DINO_MODEL', 'vits16')
    dinodir = "./../dinov3/"
    info = DINO_MODELS[model_name]
    model = torch.hub.load(dinodir, info['hub_name'], source='local', weights=info['weights'])
    model.eval()
    gpu_name, gpu_mem = detect_gpu()
    if gpu_name:
        print(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
        model = model.cuda()
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
    device = next(model.parameters()).device
    x_tensor = torch.from_numpy(x_3ch).unsqueeze(0).to(device)  # (1, 3, H, W)

    t0 = time.time()
    with torch.no_grad():
        y = model.forward_features(x_tensor)
    dt = time.time() - t0
    print(f"Inference: {dt:.3f}s for input {x_np.shape}")
    return y

def _detect_model_stride(model):
    """Detect the model's effective spatial stride by running a test input."""
    device = next(model.parameters()).device
    test_size = 128
    x_test = torch.zeros(1, 3, test_size, test_size, device=device)
    with torch.no_grad():
        y = model.forward_features(x_test)
    n_tokens = y['x_norm_patchtokens'].shape[1]
    grid_side = int(np.sqrt(n_tokens))
    model_stride = test_size // grid_side
    return model_stride

def _run_dense_passes(model, x_2d, dense_stride, model_stride):
    """Run translated passes on a pre-normalized 2D array. Returns (pH, pW, D) accumulator."""
    H, W = x_2d.shape
    pH = (H - model_stride) // dense_stride + 1
    pW = (W - model_stride) // dense_stride + 1

    embed_dim = None
    accum = None
    counts = None

    for dy in range(0, model_stride, dense_stride):
        for dx in range(0, model_stride, dense_stride):
            x_shift = x_2d[dy:, dx:]
            h_s, w_s = x_shift.shape
            h_s = (h_s // model_stride) * model_stride
            w_s = (w_s // model_stride) * model_stride
            if h_s < model_stride or w_s < model_stride:
                continue
            x_crop = x_shift[:h_s, :w_s]

            x_3ch = np.stack([x_crop] * 3)
            device = next(model.parameters()).device
            x_tensor = torch.from_numpy(x_3ch).unsqueeze(0).to(device)

            with torch.no_grad():
                y = model.forward_features(x_tensor)
            tokens = y['x_norm_patchtokens'].squeeze(0).cpu().numpy()

            if embed_dim is None:
                embed_dim = tokens.shape[1]
                accum = np.zeros((pH, pW, embed_dim), dtype=np.float64)
                counts = np.zeros((pH, pW, 1), dtype=np.float64)

            pH_s = h_s // model_stride
            pW_s = w_s // model_stride
            tokens = tokens.reshape(pH_s, pW_s, embed_dim)

            for pi in range(pH_s):
                for pj in range(pW_s):
                    oi = dy + pi * model_stride
                    oj = dx + pj * model_stride
                    gi = oi // dense_stride
                    gj = oj // dense_stride
                    if gi < pH and gj < pW:
                        accum[gi, gj] += tokens[pi, pj]
                        counts[gi, gj] += 1

    accum /= np.maximum(counts, 1)
    return accum.astype(np.float32)

def run_dino_dense(model, x_np, dense_stride=4, subtract_pos=True):
    """Dense DINO embeddings via translated passes at the model's native stride.
    If subtract_pos, subtracts a positional baseline computed from a constant image.
    Returns (pH, pW, embed_dim) numpy array at effective stride=dense_stride."""
    model_stride = _detect_model_stride(model)
    assert model_stride % dense_stride == 0, f"dense_stride={dense_stride} must divide model_stride={model_stride}"

    H, W = x_np.shape

    # Prepare input normalization
    x = x_np.astype(np.float32)
    mu, std = x.mean(), x.std() + 1e-8
    x = (x - mu) / std
    x = x * 0.229 + 0.485

    n_offsets = model_stride // dense_stride
    t0 = time.time()

    accum = _run_dense_passes(model, x, dense_stride, model_stride)

    if subtract_pos:
        x_const = np.full_like(x, 0.485)  # ImageNet mean
        baseline = _run_dense_passes(model, x_const, dense_stride, model_stride)
        accum = accum - baseline
        print(f"  Subtracted positional baseline")

    dt = time.time() - t0
    pH, pW = accum.shape[:2]
    print(f"Dense inference ({n_offsets}x{n_offsets} offsets): {dt:.1f}s for input {x_np.shape} -> {accum.shape}")
    return accum


def run_convnext_dense(model, x_np):
    """Single-pass dense embeddings for ConvNeXt models.
    Returns (pH, pW, embed_dim) numpy array."""
    model_stride = _detect_model_stride(model)
    H, W = x_np.shape

    x = x_np.astype(np.float32)
    mu, std = x.mean(), x.std() + 1e-8
    x = (x - mu) / std
    x = x * 0.229 + 0.485

    # Crop to model stride
    H = (H // model_stride) * model_stride
    W = (W // model_stride) * model_stride
    x = x[:H, :W]

    x_3ch = np.stack([x] * 3)
    device = next(model.parameters()).device
    x_tensor = torch.from_numpy(x_3ch).unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        y = model.forward_features(x_tensor)
    tokens = y['x_norm_patchtokens'].squeeze(0).cpu().numpy()

    pH = H // model_stride
    pW = W // model_stride
    token_grid = tokens.reshape(pH, pW, -1)

    dt = time.time() - t0
    print(f"ConvNeXt inference: {dt:.1f}s for input ({H},{W}) -> {token_grid.shape}")
    return token_grid

def is_convnext():
    return os.environ.get('DINO_MODEL', 'vits16').startswith('convnext')

def get_embeddings(model, x_np, dense_stride=4, subtract_pos=True):
    """Get dense embeddings. Uses translated passes for ViT, single pass for ConvNeXt."""
    if is_convnext():
        return run_convnext_dense(model, x_np)
    else:
        return run_dino_dense(model, x_np, dense_stride=dense_stride, subtract_pos=subtract_pos)

def prep_image(img, downsample_factor=1):
    """Downscale and crop to patch-aligned (multiple of 16) dimensions."""
    from skimage.transform import resize
    img = img.astype(np.float32)
    if downsample_factor > 1:
        new_h, new_w = img.shape[0] // downsample_factor, img.shape[1] // downsample_factor
        img = resize(img, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(np.float32)
    H, W = (img.shape[0] // 16) * 16, (img.shape[1] // 16) * 16
    return img[:H, :W]

def task1(n_samples=20, crop_size=1024, seed=42):
    """Download random 2D crops from liver and kidney datasets at s0.
    All crops are exactly crop_size x crop_size and fully within the volume."""
    np.random.seed(seed)

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

def task2(w, stride=16, downsample_factor=1, n_images=3, mode='nearest', dense=False, subtract_pos=True, figures_dir='figures'):
    """Run DINO on images, per-image mean subtraction, joint PCA, scrollable in napari.
    stride: patch embedding stride (ignored if dense=True).
    downsample_factor: locally downscale s0 images by this factor before running DINO.
    n_images: how many images per dataset to use.
    dense: if True, use get_embeddings (translated passes at native stride) instead of hacking stride."""
    import matplotlib.pyplot as plt
    patch_size = 16
    model = load_dino()
    if not dense:
        model.patch_embed.proj.stride = (stride, stride)

    data = load_datasets()

    for dname in ['liver', 'kidney']:
        images = data[dname]['images'][:n_images]

        # Downscale and crop to patch-aligned dims
        crops = [prep_image(img, downsample_factor) for img in images]

        H, W = crops[0].shape

        # Run DINO on all images, collect tokens + grid shapes
        all_tokens = []
        grid_shapes = []
        for i, x_crop in enumerate(crops):
            print(f"{dname} [{i}] inference...")
            if dense:
                token_grid = get_embeddings(model, x_crop, dense_stride=stride, subtract_pos=subtract_pos)  # (pH, pW, D)
                pH_i, pW_i = token_grid.shape[:2]
                tokens = token_grid.reshape(-1, token_grid.shape[2])
            else:
                y_full = run_dino(model, x_crop)
                tokens = y_full['x_norm_patchtokens'].squeeze(0).numpy()
                pH_i = (H - patch_size) // stride + 1
                pW_i = (W - patch_size) // stride + 1
            all_tokens.append(tokens)
            grid_shapes.append((pH_i, pW_i))

        # Per-image mean subtraction, then joint PCA
        for i in range(len(all_tokens)):
            all_tokens[i] = all_tokens[i] - all_tokens[i].mean(axis=0, keepdims=True)

        all_tokens_cat = np.concatenate(all_tokens, axis=0)
        pca = PCA(n_components=3)
        all_pca = pca.fit_transform(all_tokens_cat)
        print(f"{dname}: joint PCA variance = {pca.explained_variance_ratio_}")

        # Normalize globally -- RGB in [0,1]
        for c in range(3):
            lo, hi = all_pca[:, c].min(), all_pca[:, c].max()
            all_pca[:, c] = (all_pca[:, c] - lo) / (hi - lo + 1e-8)

        # Split back per image and upscale
        raw_stack = []
        pca_stack = []
        offset = 0
        for i, x_crop in enumerate(crops):
            pH_i, pW_i = grid_shapes[i]
            n = pH_i * pW_i
            pca_features = all_pca[offset:offset + n]
            offset += n

            pca_grid = pca_features.reshape(pH_i, pW_i, 3)
            pca_tensor = torch.from_numpy(pca_grid).permute(2, 0, 1).unsqueeze(0)
            ac = dict(bilinear=False, nearest=None)[mode]
            pca_img = torch.nn.functional.interpolate(pca_tensor, size=(H, W), mode=mode, align_corners=ac)
            pca_img = pca_img.squeeze(0).permute(1, 2, 0).numpy()

            raw_stack.append(x_crop)
            pca_stack.append(pca_img)

        raw_stack = np.stack(raw_stack)
        pca_stack = np.stack(pca_stack)

        label = f'{dname} {"dense" if dense else "stride"}{stride} {downsample_factor}x'
        if w is not None:
            w.add_image(raw_stack, name=f'{label} raw')
            w.add_image(pca_stack, name=f'{label} PCA', rgb=True)

        # Save tiled PCA grid as PNG
        os.makedirs(figures_dir, exist_ok=True)
        pca_grid = _tile_grid(list(pca_stack))
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(pca_grid)
        ax.set_title(f'PCA: {dname}')
        ax.axis('off')
        pca_path = os.path.join(figures_dir, f'task2_pca_{dname}.png')
        fig.savefig(pca_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  Saved {pca_path}")

        
def mitolocations():
    """Lists hold mito centerpoints for the first few images in each dataset. One point per image."""
    kidney = [
        (347,550),
        (430,400), # (388,139),
        (470,140),
        (128,68),
        (971,930),
        (220,852),
        (353,233),
    ]
    liver = [
        (619,415),
        (126,149),
        (873,159),
        (704,992),
    ]
    return {'kidney':kidney, 'liver':liver}


import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

def _deep_merge(base, override):
    """Merge override dict into base dict, recursing into sub-dicts."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged

def load_config(path=None):
    with open('config.toml', 'rb') as f:
        cfg = tomllib.load(f)
    if path is not None and path != 'config.toml':
        with open(path, 'rb') as f:
            overrides = tomllib.load(f)
        cfg = _deep_merge(cfg, overrides)
    return cfg

def run_everything(cfg=None):
    if cfg is None:
        cfg = load_config()

    g = cfg['general']
    tasks = g.get('tasks', [1, 2, 3, 4])

    # Set seeds for reproducibility
    np.random.seed(g['seed'])
    torch.manual_seed(g['seed'])

    # Set model from config
    os.environ['DINO_MODEL'] = g['model']

    if g['headless']:
        w = None
    else:
        try:
            import napari
            w = napari.viewer.Viewer()
        except ImportError:
            print("napari not installed, running without viewer")
            w = None

    print(f"Running tasks: {tasks}")

    ## Task 1: Download 20 random 1024x1024 s0 crops from liver and kidney volumes.
    if 1 in tasks:
        task1(seed=g['seed'])

    ## Task 2: Dense DINO embeddings + joint PCA visualized as RGB across multiple images.
    if 2 in tasks:
        t2 = cfg['task2']
        task2(w, stride=g['stride'], downsample_factor=g['downsample_factor'],
              n_images=t2['n_images'], dense=t2['dense'], subtract_pos=g['subtract_pos'],
              figures_dir=g['figures_dir'])

    ## Task 3: Mito retrieval — average cosine similarity from query points across targets.
    if 3 in tasks:
        t3 = cfg['task3']
        mitos = mitolocations()
        query_idxs = list(range(len(mitos[t3['query_ds']])))
        task3(w, query_ds=t3['query_ds'], target_ds=t3['target_ds'],
              query_idxs=query_idxs, n_targets=t3['n_targets'],
              downsample_factor=g['downsample_factor'], dense_stride=g['stride'])

    ## Task 4: Save tiled grids for all query/target combinations.
    if 4 in tasks:
        task3_all(dense_stride=g['stride'], downsample_factor=g['downsample_factor'],
                  n_targets=cfg['task3']['n_targets'], out_dir=g['figures_dir'])

    ## Task 5: Compare all available models on one image from each dataset.
    if 5 in tasks:
        task5(downsample_factor=g['downsample_factor'], dense_stride=g['stride'],
              subtract_pos=g['subtract_pos'], figures_dir=g['figures_dir'])


def task3(w, query_ds='kidney', target_ds='kidney', query_idxs=None, n_targets=3,
         dense_stride=8, downsample_factor=1):
    """Embedding-based retrieval: average query from multiple mito points, predict on target images.
    query_ds/target_ds: 'liver' or 'kidney'.
    query_idxs: which mito points to use as queries (indices into mitolocations). None = all.
    n_targets: how many target images to predict on.
    Results are collected into scrollable stacks."""
    patch_size = 16

    model = load_dino()
    data = load_datasets()
    mitos = mitolocations()

    # --- Select query points ---
    if query_idxs is None:
        query_idxs = list(range(len(mitos[query_ds])))
    query_points = [mitos[query_ds][i] for i in query_idxs]

    # --- Collect query embeddings (normalized, not averaged yet) ---
    query_embs = []  # list of (1, D) normalized tensors
    for i, mito_yx in zip(query_idxs, query_points):
        x_crop = prep_image(data[query_ds]['images'][i], downsample_factor)

        print(f"Query {query_ds}[{i}]: computing embeddings...")
        token_grid = get_embeddings(model, x_crop, dense_stride=dense_stride)
        qy, qx = mito_yx[0] // downsample_factor, mito_yx[1] // downsample_factor
        pH, pW = token_grid.shape[:2]
        qi = min(qy // dense_stride, pH - 1)
        qj = min(qx // dense_stride, pW - 1)
        qvec = torch.from_numpy(token_grid[qi, qj]).unsqueeze(0)
        query_embs.append(torch.nn.functional.normalize(qvec, dim=-1))

    print(f"Using {len(query_embs)} query points from {query_ds}")

    # --- Compute per-query similarity on each target, then average the similarity maps ---
    target_images = data[target_ds]['images'][:n_targets]
    raw_stack = []
    sim_stack = []

    for i, img in enumerate(target_images):
        x_crop = prep_image(img, downsample_factor)

        print(f"Target {target_ds}[{i}]: computing embeddings...")
        token_grid = get_embeddings(model, x_crop, dense_stride=dense_stride)
        pH, pW, D = token_grid.shape

        tokens_flat = torch.from_numpy(token_grid.reshape(-1, D))
        tokens_normed = torch.nn.functional.normalize(tokens_flat, dim=-1)

        # Compute cosine similarity for each query, then average
        sim_accum = np.zeros(tokens_normed.shape[0], dtype=np.float64)
        for qemb in query_embs:
            sim_accum += (tokens_normed @ qemb.T).squeeze(-1).numpy()
        sim_accum /= len(query_embs)

        sim_grid = sim_accum.reshape(pH, pW)
        H, W = x_crop.shape
        sim_tensor = torch.from_numpy(sim_grid).float().unsqueeze(0).unsqueeze(0)
        sim_img = torch.nn.functional.interpolate(sim_tensor, size=(H, W), mode='bilinear', align_corners=False)
        sim_img = sim_img.squeeze().numpy()

        raw_stack.append(x_crop)
        sim_stack.append(sim_img)

    raw_stack = np.stack(raw_stack)
    sim_stack = np.stack(sim_stack)

    if w is not None:
        label = f'q={query_ds}[{query_idxs}] t={target_ds}'
        w.add_image(raw_stack, name=f'{label} raw')
        w.add_image(sim_stack, name=f'{label} sim', colormap='inferno')

    return raw_stack, sim_stack

def _tile_grid(images, ncols=None):
    """Tile a list of 2D images into a single grid image."""
    n = len(images)
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    H, W = images[0].shape[:2]
    extra = images[0].shape[2:] # () for grayscale, (3,) for RGB
    grid = np.zeros((nrows * H, ncols * W) + extra, dtype=images[0].dtype)
    for idx, img in enumerate(images):
        r, c = divmod(idx, ncols)
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = img
    return grid

def task3_all(dense_stride=8, downsample_factor=2, n_targets=None, out_dir='figures'):
    """Run task3 for all 4 query/target combinations and save tiled grids as PNGs.
    n_targets: number of target images per combo. None = all available."""
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    mitos = mitolocations()
    data = load_datasets()
    combos = [
        ('kidney', 'kidney'),
        ('kidney', 'liver'),
        ('liver', 'liver'),
        ('liver', 'kidney'),
    ]

    for query_ds, target_ds in combos:
        query_idxs = list(range(len(mitos[query_ds])))
        nt = n_targets if n_targets is not None else len(data[target_ds]['images'])
        print(f"\n=== q={query_ds} t={target_ds} ({len(query_idxs)} queries, {nt} targets) ===")
        raw_stack, sim_stack = task3(
            w=None, query_ds=query_ds, target_ds=target_ds,
            query_idxs=query_idxs, n_targets=nt,
            dense_stride=dense_stride, downsample_factor=downsample_factor,
        )

        # Tile raw and sim grids
        raw_grid = _tile_grid(list(raw_stack))
        sim_grid = _tile_grid(list(sim_stack))

        # Normalize sim: clamp lower bound, scale to [0,1], apply gamma
        # lb and gamma determined by visual inspection
        sim_norm = (sim_grid - 0.533) / (sim_grid.max() - 0.533 + 1e-8)
        sim_norm = np.clip(sim_norm, 0, 1)
        sim_norm = sim_norm ** (1.0 / 1.8)

        # Save raw grid
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(raw_grid, cmap='gray')
        ax.set_title(f'Raw: query={query_ds}, target={target_ds}')
        ax.axis('off')
        raw_path = os.path.join(out_dir, f'task3_raw_q{query_ds}_t{target_ds}.png')
        fig.savefig(raw_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        # Save sim grid
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(sim_norm, cmap='inferno')
        ax.set_title(f'Similarity: query={query_ds}, target={target_ds}')
        ax.axis('off')
        sim_path = os.path.join(out_dir, f'task3_sim_q{query_ds}_t{target_ds}.png')
        fig.savefig(sim_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        # Combine raw and sim into an oscillating GIF
        from PIL import Image
        raw_pil = Image.open(raw_path)
        sim_pil = Image.open(sim_path)
        gif_path = os.path.join(out_dir, f'task3_q{query_ds}_t{target_ds}.gif')
        raw_pil.save(gif_path, save_all=True, append_images=[sim_pil],
                     duration=1000, loop=0)

        print(f"  Saved {raw_path}")
        print(f"  Saved {sim_path}")
        print(f"  Saved {gif_path}")

def task5(downsample_factor=2, dense_stride=4, subtract_pos=True, figures_dir='figures'):
    """Compare all available models on one image from each dataset.
    Saves a PCA RGB PNG for each (model, dataset) pair."""
    import matplotlib.pyplot as plt
    os.makedirs(figures_dir, exist_ok=True)

    data = load_datasets()

    # Find which models have weights on disk
    available = {k: v for k, v in DINO_MODELS.items() if os.path.exists(v['weights'])}
    print(f"Available models: {list(available.keys())}")

    for model_name in available:
        print(f"\n=== Loading {model_name} ===")
        os.environ['DINO_MODEL'] = model_name
        model = load_dino(model_name)

        for dname in ['liver', 'kidney']:
            img = data[dname]['images'][0]
            x_crop = prep_image(img, downsample_factor)
            H, W = x_crop.shape

            print(f"  {dname}: inference...")
            token_grid = get_embeddings(model, x_crop, dense_stride=dense_stride, subtract_pos=subtract_pos)
            pH, pW = token_grid.shape[:2]
            tokens = token_grid.reshape(-1, token_grid.shape[2])

            # Per-image mean subtraction + PCA
            tokens = tokens - tokens.mean(axis=0, keepdims=True)
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(tokens)
            for c in range(3):
                lo, hi = pca_features[:, c].min(), pca_features[:, c].max()
                pca_features[:, c] = (pca_features[:, c] - lo) / (hi - lo + 1e-8)

            pca_grid = pca_features.reshape(pH, pW, 3)
            pca_tensor = torch.from_numpy(pca_grid).permute(2, 0, 1).unsqueeze(0)
            pca_img = torch.nn.functional.interpolate(pca_tensor, size=(H, W), mode='bilinear', align_corners=False)
            pca_img = pca_img.squeeze(0).permute(1, 2, 0).numpy()

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(x_crop, cmap='gray')
            axes[0].set_title(f'{dname} raw')
            axes[0].axis('off')
            axes[1].imshow(pca_img)
            axes[1].set_title(f'{model_name} PCA')
            axes[1].axis('off')
            path = os.path.join(figures_dir, f'task5_{model_name}_{dname}.png')
            fig.savefig(path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"  Saved {path}")

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.toml', help='Path to config file')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Auto-detect: if big GPU available, default to headless with large model
    gpu_name, gpu_mem = detect_gpu()
    if gpu_name and gpu_mem >= 40:
        print(f"Detected large GPU: {gpu_name} ({gpu_mem:.0f} GB)")
        cfg['general']['headless'] = True
        if os.path.exists(DINO_MODELS['vitl16']['weights']):
            cfg['general']['model'] = 'vitl16'
            print("Auto-selecting vitl16 model")

    print(f"Config: model={cfg['general']['model']}, stride={cfg['general']['stride']}, "
          f"downsample={cfg['general']['downsample_factor']}, headless={cfg['general']['headless']}")
    run_everything(cfg)
