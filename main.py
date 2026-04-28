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

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

print(f"Imports done in {time.time() - _t_start:.1f}s")

# Inferno colormap LUT (256 entries, RGB uint8) — avoids matplotlib dependency
_INFERNO_LUT = np.array([
    [0,0,4],[1,0,5],[1,1,6],[1,1,8],[2,1,10],[2,2,12],[2,2,14],[3,2,16],
    [4,3,18],[4,3,20],[5,4,23],[6,4,25],[7,5,27],[8,5,29],[9,6,31],[10,7,34],
    [11,7,36],[12,8,38],[13,8,41],[14,9,43],[16,9,45],[17,10,48],[18,10,50],
    [20,11,52],[21,11,55],[22,11,57],[24,12,60],[25,12,62],[27,12,65],[28,12,67],
    [30,12,69],[31,12,72],[33,12,74],[35,12,76],[36,12,79],[38,12,81],[40,11,83],
    [41,11,85],[43,11,87],[45,11,89],[47,10,91],[49,10,92],[50,10,94],[52,10,95],
    [54,9,97],[56,9,98],[57,9,99],[59,9,100],[61,9,101],[62,9,102],[64,10,103],
    [66,10,104],[68,10,104],[69,10,105],[71,11,106],[73,11,106],[74,12,107],
    [76,12,107],[77,13,108],[79,13,108],[81,14,108],[82,14,109],[84,15,109],
    [85,15,109],[87,16,110],[89,16,110],[90,17,110],[92,18,110],[93,18,110],
    [95,19,110],[97,19,110],[98,20,110],[100,21,110],[101,21,110],[103,22,110],
    [105,22,110],[106,23,110],[108,24,110],[109,24,110],[111,25,110],[113,25,110],
    [114,26,110],[116,26,110],[117,27,110],[119,28,109],[120,28,109],[122,29,109],
    [124,29,109],[125,30,109],[127,30,108],[128,31,108],[130,32,108],[132,32,107],
    [133,33,107],[135,33,107],[136,34,106],[138,34,106],[140,35,105],[141,35,105],
    [143,36,105],[144,37,104],[146,37,104],[148,38,103],[149,38,103],[151,39,102],
    [152,39,102],[154,40,101],[156,40,100],[157,41,100],[159,41,99],[160,42,99],
    [162,42,98],[164,43,97],[165,44,96],[167,44,96],[168,45,95],[170,45,94],
    [172,46,93],[173,46,93],[175,47,92],[176,48,91],[178,48,90],[179,49,89],
    [181,49,88],[182,50,88],[184,51,87],[185,51,86],[187,52,85],[188,53,84],
    [190,53,83],[191,54,82],[193,55,81],[194,55,80],[196,56,79],[197,57,78],
    [198,58,77],[200,58,76],[201,59,75],[203,60,74],[204,61,73],[205,62,72],
    [207,62,71],[208,63,70],[209,64,69],[211,65,68],[212,66,67],[213,67,66],
    [214,68,65],[216,69,63],[217,70,62],[218,71,61],[219,72,60],[220,73,59],
    [221,74,58],[222,75,56],[224,76,55],[225,77,54],[226,78,53],[227,79,52],
    [228,81,51],[229,82,49],[230,83,48],[231,84,47],[232,85,46],[232,87,44],
    [233,88,43],[234,89,42],[235,90,41],[236,92,39],[236,93,38],[237,94,37],
    [238,96,36],[238,97,34],[239,98,33],[240,100,32],[240,101,31],[241,103,29],
    [241,104,28],[242,105,27],[242,107,25],[243,108,24],[243,110,23],[243,111,22],
    [244,113,20],[244,114,19],[244,116,18],[245,117,17],[245,119,15],[245,120,14],
    [245,122,13],[246,123,12],[246,125,11],[246,126,10],[246,128,9],[247,130,8],
    [247,131,7],[247,133,6],[247,134,5],[247,136,5],[247,137,4],[248,139,4],
    [248,141,3],[248,142,3],[248,144,3],[248,145,3],[248,147,3],[248,148,3],
    [249,150,3],[249,151,4],[249,153,4],[249,154,5],[249,156,6],[249,157,7],
    [249,159,8],[249,160,9],[249,162,10],[249,163,12],[249,165,13],[249,166,15],
    [249,168,16],[249,169,18],[249,171,20],[249,172,21],[249,174,23],[249,175,25],
    [249,177,27],[249,178,29],[248,180,31],[248,181,33],[248,183,35],[248,184,37],
    [248,186,39],[247,187,41],[247,189,43],[247,190,46],[247,192,48],[246,193,50],
    [246,195,52],[246,196,55],[245,198,57],[245,199,59],[245,201,62],[244,202,64],
    [244,204,66],[244,205,69],[243,207,71],[243,208,74],[243,210,76],[242,211,79],
    [242,213,81],[242,214,84],[241,216,87],[241,217,89],[241,219,92],[240,220,95],
    [240,222,97],[240,223,100],[239,225,103],[239,226,106],[239,228,108],
    [239,229,111],[238,231,114],[238,232,117],[238,234,120],[238,235,123],
    [237,237,126],[237,238,129],[237,240,132],[237,241,135],[237,243,138],
    [236,244,141],[236,246,144],[236,247,147],[236,249,150],[237,250,154],
    [237,252,157],[237,253,160],[238,255,163],
], dtype=np.uint8)

def _apply_colormap(arr):
    """Apply inferno colormap to a [0,1] float array. Returns (H, W, 3) uint8."""
    indices = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return _INFERNO_LUT[indices]

CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)
print("Ensured CACHE dir.")

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
    'vitl16': {
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
    'vitl16_sat': {
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

def load_dino(model_name='vits16'):
    print(f"Loading model {model_name}...")
    t0 = time.time()
    dinodir = "./../dinov3/"
    info = DINO_MODELS[model_name]
    model = torch.hub.load(dinodir, info['hub_name'], source='local', pretrained=False)
    state_dict = torch.load(info['weights'], map_location='cpu', weights_only=True)
    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        print(f"  Ignored unexpected keys: {result.unexpected_keys}")
    model.eval()
    gpu_name, gpu_mem = detect_gpu()
    if gpu_name:
        model = model.cuda()
        print(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    return model

_dataset_stats = {}  # computed on first call to _compute_dataset_stats()

def _compute_dataset_stats():
    """Compute mean and std across all images in each dataset. Cached."""
    if _dataset_stats:
        return _dataset_stats
    data = load_datasets()
    for dname in data:
        all_pixels = np.concatenate([img.ravel().astype(np.float64) for img in data[dname]['images']])
        _dataset_stats[dname] = (float(all_pixels.mean()), float(all_pixels.std()))
        print(f"  {dname} stats: mean={_dataset_stats[dname][0]:.1f}, std={_dataset_stats[dname][1]:.1f}")
    return _dataset_stats

def _normalize_gray_to_3ch(x_np, dataset='liver'):
    """Convert grayscale (H,W) to zero-mean unit-variance (3,H,W) float32 using dataset stats."""
    stats = _compute_dataset_stats()
    mean, std = stats[dataset]
    x = (x_np.astype(np.float32) - mean) / (std + 1e-8)
    return np.stack([x] * 3)  # (3, H, W)

def run_dino(model, x_np, dataset='liver'):
    """Run DINO on a 2D grayscale numpy array. H and W must be divisible by 16."""
    x_3ch = _normalize_gray_to_3ch(x_np, dataset)
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

def _run_dense_passes(model, x_3ch, dense_stride, model_stride):
    """Run translated passes on a pre-normalized (3,H,W) array. Returns (pH, pW, D) accumulator."""
    _, H, W = x_3ch.shape
    pH = (H - model_stride) // dense_stride + 1
    pW = (W - model_stride) // dense_stride + 1

    embed_dim = None
    accum = None
    counts = None

    for dy in range(0, model_stride, dense_stride):
        for dx in range(0, model_stride, dense_stride):
            x_shift = x_3ch[:, dy:, dx:]
            _, h_s, w_s = x_shift.shape
            h_s = (h_s // model_stride) * model_stride
            w_s = (w_s // model_stride) * model_stride
            if h_s < model_stride or w_s < model_stride:
                continue
            x_crop = x_shift[:, :h_s, :w_s]

            device = next(model.parameters()).device
            x_tensor = torch.from_numpy(x_crop).unsqueeze(0).to(device)

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

def run_dino_dense(model, x_np, dense_stride=4, subtract_pos=True, dataset='liver'):
    """Dense DINO embeddings via translated passes at the model's native stride.
    If subtract_pos, subtracts a positional baseline computed from a constant image.
    Returns (pH, pW, embed_dim) numpy array at effective stride=dense_stride."""
    model_stride = _detect_model_stride(model)
    assert model_stride % dense_stride == 0, f"dense_stride={dense_stride} must divide model_stride={model_stride}"

    H, W = x_np.shape

    x = _normalize_gray_to_3ch(x_np, dataset)  # (3, H, W)

    n_offsets = model_stride // dense_stride
    t0 = time.time()

    accum = _run_dense_passes(model, x, dense_stride, model_stride)

    if subtract_pos:
        x_const = np.zeros_like(x)  # zero after normalization = dataset mean
        baseline = _run_dense_passes(model, x_const, dense_stride, model_stride)
        accum = accum - baseline
        print(f"  Subtracted positional baseline")

    dt = time.time() - t0
    pH, pW = accum.shape[:2]
    print(f"Dense inference ({n_offsets}x{n_offsets} offsets): {dt:.1f}s for input {x_np.shape} -> {accum.shape}")
    return accum


def run_convnext_dense(model, x_np, dataset='liver'):
    """Single-pass dense embeddings for ConvNeXt models.
    Returns (pH, pW, embed_dim) numpy array."""
    model_stride = _detect_model_stride(model)
    H, W = x_np.shape

    # Crop to model stride
    H = (H // model_stride) * model_stride
    W = (W // model_stride) * model_stride

    x_3ch = _normalize_gray_to_3ch(x_np[:H, :W], dataset)
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

def get_embeddings(model, x_np, dense_stride=4, subtract_pos=True, model_name='vits16', dataset='liver'):
    """Get dense embeddings. Uses translated passes for ViT, single pass for ConvNeXt."""
    if model_name.startswith('convnext'):
        return run_convnext_dense(model, x_np, dataset=dataset)
    else:
        return run_dino_dense(model, x_np, dense_stride=dense_stride, subtract_pos=subtract_pos, dataset=dataset)

def prep_image(img, downsample_factor=1):
    """Downscale and crop to patch-aligned (multiple of 16) dimensions."""
    img = img.astype(np.float32)
    ## FIX: what about downsample_factor < 1 ?
    if downsample_factor > 1:
        new_h, new_w = img.shape[0] // downsample_factor, img.shape[1] // downsample_factor
        img = resize(img, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(np.float32)
    H, W = (img.shape[0] // 16) * 16, (img.shape[1] // 16) * 16
    return img[:H, :W]

def task1(cfg, n_samples=20, crop_size=1024):
    """Download random 2D crops from liver and kidney datasets at s0."""
    np.random.seed(cfg.general.seed)

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

SKIP_IMAGES = {
    'kidney': [4],  # image 4 has a large black region
}

def load_datasets():
    """Load all task1 crops into a dict keyed by dataset name.
    Returns {'liver': {'images': [np arrays], 'coords': [(z,y,x), ...]},
             'kidney': {...}}"""
    result = {}
    for dname in ['liver', 'kidney']:
        out_dir = os.path.join(CACHE_DIR, f'task1_{dname}')
        assert os.path.isdir(out_dir), f"No data found at {out_dir}. Run task1() first."
        files = sorted([f for f in os.listdir(out_dir) if f.endswith('.npy')])
        skip = SKIP_IMAGES.get(dname, [])
        images = []
        coords = []
        for idx, f in enumerate(files):
            if idx in skip:
                print(f"  Skipping {dname}[{idx}]: {f}")
                continue
            parts = f.replace('.npy', '').split('_')
            z = int(parts[1][1:])
            y = int(parts[2][1:])
            x = int(parts[3][1:])
            images.append(np.load(os.path.join(out_dir, f)))
            coords.append((z, y, x))
        result[dname] = {'images': images, 'coords': coords}
        print(f"{dname}: loaded {len(images)} images, shape={images[0].shape}")
    return result

def task2(cfg):
    """Run DINO on images, per-image mean subtraction, joint PCA."""

    g = cfg.general
    t2 = cfg.task2
    model = load_dino(g.model)

    data = load_datasets()

    for dname in ['liver', 'kidney']:
        images = data[dname]['images'][:t2.n_images]
        crops = [prep_image(img, g.downsample_factor) for img in images]
        H, W = crops[0].shape

        all_tokens = []
        grid_shapes = []
        for i, x_crop in enumerate(crops):
            print(f"{dname} [{i}] inference...")
            token_grid = get_embeddings(model, x_crop, dense_stride=g.stride, subtract_pos=g.subtract_pos, model_name=g.model, dataset=dname)
            pH_i, pW_i = token_grid.shape[:2]
            tokens = token_grid.reshape(-1, token_grid.shape[2])
            all_tokens.append(tokens)
            grid_shapes.append((pH_i, pW_i))

        # Per-image mean subtraction, then joint PCA
        # for i in range(len(all_tokens)):
        #     all_tokens[i] = all_tokens[i] - all_tokens[i].mean(axis=0, keepdims=True)

        all_tokens_cat = np.concatenate(all_tokens, axis=0)
        pca = PCA(n_components=3)
        all_pca = pca.fit_transform(all_tokens_cat)
        print(f"{dname}: joint PCA variance = {pca.explained_variance_ratio_} (total={pca.explained_variance_ratio_.sum():.3f})")

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
            pca_img = torch.nn.functional.interpolate(pca_tensor, size=(H, W), mode='bilinear', align_corners=False)
            pca_img = pca_img.squeeze(0).permute(1, 2, 0).numpy()

            raw_stack.append(x_crop)
            pca_stack.append(pca_img)

        raw_stack = np.stack(raw_stack)
        pca_stack = np.stack(pca_stack)

        os.makedirs(g.figures_dir, exist_ok=True)

        raw_grid = _tile_grid(list(raw_stack))
        pca_grid = _tile_grid(list(pca_stack))
        raw_pil = Image.fromarray(raw_grid.astype(np.uint8))
        pca_pil = Image.fromarray((pca_grid * 255).astype(np.uint8))

        raw_path = os.path.join(g.figures_dir, f'task2_raw_{dname}.png')
        pca_path = os.path.join(g.figures_dir, f'task2_pca_{dname}.png')
        gif_path = os.path.join(g.figures_dir, f'task2_{dname}.gif')
        raw_pil.save(raw_path)
        pca_pil.save(pca_path)
        raw_pil.save(gif_path, save_all=True, append_images=[pca_pil], duration=1000, loop=0)
        print(f"  Saved {raw_path}, {pca_path}, {gif_path}")

        
def mitolocations():
    """Lists hold mito centerpoints for the first few images in each dataset. One point per image."""
    kidney = [
        (347,550),
        # (430,400), # (388,139),
        # (470,140),
        # (128,68),
        # (971,930),
        # (220,852),
        # (353,233),
    ]
    liver = [
        (619,415),
        # (126,149),
        # (873,159),
        # (704,992),
    ]
    return {'kidney':kidney, 'liver':liver}


def _deep_merge(base, override):
    """Merge override dict into base dict, recursing into sub-dicts."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged

def _dict_to_ns(d):
    """Recursively convert a dict to nested SimpleNamespace."""
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_ns(v))
        else:
            setattr(ns, k, v)
    return ns

def load_config(path=None):
    with open('config.toml', 'rb') as f:
        cfg = tomllib.load(f)
    if path is not None and path != 'config.toml':
        with open(path, 'rb') as f:
            overrides = tomllib.load(f)
        cfg = _deep_merge(cfg, overrides)
    return _dict_to_ns(cfg)

def run_everything(cfg=None):
    if cfg is None:
        cfg = load_config()

    g = cfg.general
    tasks = getattr(g, 'tasks', [1, 2, 3, 4])

    # Set seeds for reproducibility
    np.random.seed(g.seed)
    torch.manual_seed(g.seed)

    # Clean figures directory
    import shutil
    if os.path.exists(g.figures_dir):
        shutil.rmtree(g.figures_dir)
    os.makedirs(g.figures_dir)

    print(f"Running tasks: {tasks}")

    if 1 in tasks: task1(cfg)
    if 2 in tasks: task2(cfg)
    if 3 in tasks: task3(cfg)
    if 4 in tasks: task4(cfg)


def _retrieval(model, query_ds, target_ds, n_targets, downsample_factor=2, dense_stride=2, model_name='vits16', subtract_pos=True):
    """Embedding-based retrieval helper. Returns (raw_stack, sim_stack)."""
    data = load_datasets()
    mitos = mitolocations()
    query_points = mitos[query_ds]

    query_embs = []
    for i, mito_yx in enumerate(query_points):
        x_crop = prep_image(data[query_ds]['images'][i], downsample_factor)

        print(f"Query {query_ds}[{i}]: computing embeddings...")
        token_grid = get_embeddings(model, x_crop, dense_stride=dense_stride, subtract_pos=subtract_pos, model_name=model_name, dataset=query_ds)

        # Per-image mean subtraction
        tokens = token_grid.reshape(-1, token_grid.shape[2])
        tokens = tokens - tokens.mean(axis=0, keepdims=True)
        token_grid = tokens.reshape(token_grid.shape)

        # Scale mito x,y coords to match image scale
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
        token_grid = get_embeddings(model, x_crop, dense_stride=dense_stride, subtract_pos=subtract_pos, model_name=model_name, dataset=target_ds)
        pH, pW, D = token_grid.shape

        # Per-image mean subtraction
        tokens_flat = token_grid.reshape(-1, D)
        tokens_flat = tokens_flat - tokens_flat.mean(axis=0, keepdims=True)
        tokens_flat = torch.from_numpy(tokens_flat)
        tokens_normed = torch.nn.functional.normalize(tokens_flat, dim=-1)

        # Compute cosine similarity for each query, then average
        sim_accum = np.zeros(tokens_normed.shape[0], dtype=np.float64)
        # FIX: this could be a single line like:
        # sim_accum = (tokens_normed @ qemb.T).mean(-1).numpy()
        for qemb in query_embs:
            sim_accum += (tokens_normed @ qemb.T).squeeze(-1).numpy()
        sim_accum /= len(query_embs)

        sim_grid = sim_accum.reshape(pH, pW)
        H, W = x_crop.shape
        sim_tensor = torch.from_numpy(sim_grid).float().unsqueeze(0).unsqueeze(0)
        sim_img = torch.nn.functional.interpolate(sim_tensor, size=(H, W), mode='bilinear', align_corners=False)
        sim_img = sim_img.squeeze(0).squeeze(0).numpy()

        raw_stack.append(x_crop)
        sim_stack.append(sim_img)

    raw_stack = np.stack(raw_stack)
    sim_stack = np.stack(sim_stack)

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

# def normaffine01(x, mi, ma):
#     x = (x - mi) / (ma - mi)
    

def task3(cfg):
    """Run retrieval for all 4 query/target combinations and save tiled grids as PNGs."""

    g = cfg.general
    out_dir = g.figures_dir
    os.makedirs(out_dir, exist_ok=True)

    model = load_dino(g.model)
    mitos = mitolocations()
    data = load_datasets()
    combos = [
        ('kidney', 'kidney'),
        ('kidney', 'liver'),
        ('liver', 'liver'),
        ('liver', 'kidney'),
    ]

    for query_ds, target_ds in combos:
        nt = cfg.task3.n_targets if cfg.task3.n_targets else len(data[target_ds]['images'])
        n_queries = len(mitos[query_ds])
        print(f"\n=== q={query_ds} t={target_ds} ({n_queries} queries, {nt} targets) ===")
        raw_stack, sim_stack = _retrieval(
            model, query_ds=query_ds, target_ds=target_ds, n_targets=nt,
            downsample_factor=g.downsample_factor, dense_stride=g.stride,
            model_name=g.model, subtract_pos=g.subtract_pos,
        )

        raw_grid = _tile_grid(list(raw_stack))
        sim_norm = (sim_stack - sim_stack.min()) / (sim_stack.max() - sim_stack.min() + 1e-8)
        sim_grid = _tile_grid(list(sim_norm))

        raw_pil = Image.fromarray(raw_grid.astype(np.uint8))
        sim_pil = Image.fromarray(_apply_colormap(sim_grid))

        raw_path = os.path.join(out_dir, f'task3_raw_q{query_ds}_t{target_ds}.png')
        sim_path = os.path.join(out_dir, f'task3_sim_q{query_ds}_t{target_ds}.png')
        gif_path = os.path.join(out_dir, f'task3_q{query_ds}_t{target_ds}.gif')
        raw_pil.save(raw_path)
        sim_pil.save(sim_path)
        raw_pil.save(gif_path, save_all=True, append_images=[sim_pil], duration=1000, loop=0)
        print(f"  Saved {raw_path}, {sim_path}, {gif_path}")

def task4(cfg):
    """Compare all available models on one image from each dataset.
    Saves a PCA RGB PNG for each (model, dataset) pair."""

    g = cfg.general
    os.makedirs(g.figures_dir, exist_ok=True)

    data = load_datasets()

    available = {k: v for k, v in DINO_MODELS.items() if os.path.exists(v['weights'])}
    print(f"Available models: {list(available.keys())}")

    for model_name in available:
        print(f"\n=== Loading {model_name} ===")
        model = load_dino(model_name)

        for dname in ['liver', 'kidney']:
            img = data[dname]['images'][0]
            x_crop = prep_image(img, g.downsample_factor)
            H, W = x_crop.shape

            print(f"  {dname}: inference...")
            token_grid = get_embeddings(model, x_crop, dense_stride=g.stride, subtract_pos=g.subtract_pos, model_name=model_name, dataset=dname)
            pH, pW = token_grid.shape[:2]
            tokens = token_grid.reshape(-1, token_grid.shape[2])

            # Per-image mean subtraction + PCA
            tokens = tokens - tokens.mean(axis=0, keepdims=True)
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(tokens)
            print(f"    PCA variance = {pca.explained_variance_ratio_} (total={pca.explained_variance_ratio_.sum():.3f})")
            for c in range(3):
                lo, hi = pca_features[:, c].min(), pca_features[:, c].max()
                pca_features[:, c] = (pca_features[:, c] - lo) / (hi - lo + 1e-8)

            pca_grid = pca_features.reshape(pH, pW, 3)
            pca_tensor = torch.from_numpy(pca_grid).permute(2, 0, 1).unsqueeze(0)
            pca_img = torch.nn.functional.interpolate(pca_tensor, size=(H, W), mode='bilinear', align_corners=False)
            pca_img = pca_img.squeeze(0).permute(1, 2, 0).numpy()

            raw_pil = Image.fromarray(x_crop.astype(np.uint8))
            pca_pil = Image.fromarray((pca_img * 255).astype(np.uint8))

            raw_path = os.path.join(g.figures_dir, f'task4_raw_{model_name}_{dname}.png')
            pca_path = os.path.join(g.figures_dir, f'task4_pca_{model_name}_{dname}.png')
            gif_path = os.path.join(g.figures_dir, f'task4_{model_name}_{dname}.gif')
            raw_pil.save(raw_path)
            pca_pil.save(pca_path)
            raw_pil.save(gif_path, save_all=True, append_images=[pca_pil], duration=1000, loop=0)
            print(f"  Saved {raw_path}, {pca_path}, {gif_path}")

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.toml', help='Path to config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("config = ", cfg)
    run_everything(cfg)
