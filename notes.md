# Notebook

The `jrc_hela-3` data doesn't look good. It's hard to visually identify distinct mitos.
Let's target smaller mitos that are closer to the DINOv3 patch size and let's look for two
distinct datasets.

I'm pulling down data from `s3://janelia-cosem-datasets` at about 5s / MB. Isn't this slow?
And 32MB for one z slice of hela-2 = 3m download! We can try randomly sampling inside each
volume, but it's going to be painful. It looks like most datasets come pre-pooled at various
resolutions; first we need to find the right res. Let's start off with a single slice and
see if we can run DINOv3 on it.

NB: Structures from neiboring z slices will share features more than different mito from
same volume. When sampling patches we should choose a sampling method that gives both
within- and across- mito samples. We can assume categories based on patch location.

FIX: The DINOv3 download links don't work with wget or curl! It's a mystery...
Let's start off with the smallest model 21M.

OK, the RGB plot of top 3 PCA dims reveal a very weak correlation with image content and a
very strong correlation with spatial position! So either
1. there's something wrong with our rope embeddings
2. there's something wrong with our image input norm
3. the image resolution is so wrong that the model is producing random noise

Let's do the same RGB-PCA plot for a few samples of random noise...



# Questions

Q: How are we going to validate clusters without supervised GT?
A: Some datasets DO have GT.

# shapes

//janelia-cosem-datasets/jrc_fly-larva-1/jrc_fly-larva-1.n5
/Users/broaddus/Desktop/janelia-task/gets3data.py:31: FutureWarning: The N5FSStore is deprecated and will be removed in a Zarr-Python version 3, see https://github.com/zarr-developers/zarr-python/issues/1274 and https://github.com/zarr-developers/n5py for more information.
  group = zarr.open(zarr.N5FSStore(path, anon=True)) # access the root of the n5 container
/ <zarr.hierarchy.Group '/shape'> <zarr.hierarchy.Group '/dtype'>
 └── em <zarr.hierarchy.Group '/em/shape'> <zarr.hierarchy.Group '/em/dtype'>
     └── tem-uint8 <zarr.hierarchy.Group '/em/tem-uint8/shape'> <zarr.hierarchy.Group '/em/tem-uint8/dtype'>
         ├── s0 (4816, 31616, 99840) uint8
         ├── s1 (4816, 15808, 49920) uint8
         ├── s2 (4816, 7904, 24960) uint8
         ├── s3 (4816, 3952, 12480) uint8
         ├── s4 (4816, 1976, 6240) uint8
         ├── s5 (4816, 988, 3120) uint8
         ├── s6 (4816, 494, 1560) uint8
         ├── s7 (4816, 247, 780) uint8
         └── s8 (4816, 123, 390) uint8
s3://janelia-cosem-datasets/jrc_fly-larva-1/jrc_fly-larva-1.n5
/Users/broaddus/Desktop/janelia-task/gets3data.py:31: FutureWarning: The N5FSStore is deprecated and will be removed in a Zarr-Python version 3, see https://github.com/zarr-developers/zarr-python/issues/1274 and https://github.com/zarr-developers/n5py for more information.
  group = zarr.open(zarr.N5FSStore(path, anon=True)) # access the root of the n5 container
/ <zarr.hierarchy.Group '/shape'> <zarr.hierarchy.Group '/dtype'>
 └── em <zarr.hierarchy.Group '/em/shape'> <zarr.hierarchy.Group '/em/dtype'>
     └── tem-uint8 <zarr.hierarchy.Group '/em/tem-uint8/shape'> <zarr.hierarchy.Group '/em/tem-uint8/dtype'>
         ├── s0 (4816, 31616, 99840) uint8
         ├── s1 (4816, 15808, 49920) uint8
         ├── s2 (4816, 7904, 24960) uint8
         ├── s3 (4816, 3952, 12480) uint8
         ├── s4 (4816, 1976, 6240) uint8
         ├── s5 (4816, 988, 3120) uint8
         ├── s6 (4816, 494, 1560) uint8
         ├── s7 (4816, 247, 780) uint8
         └── s8 (4816, 123, 390) uint8
s3://janelia-cosem-datasets/jrc_jurkat-1/jrc_jurkat-1.n5
/ <zarr.hierarchy.Group '/shape'> <zarr.hierarchy.Group '/dtype'>
 ├── em <zarr.hierarchy.Group '/em/shape'> <zarr.hierarchy.Group '/em/dtype'>
 │   └── fibsem-uint16 <zarr.hierarchy.Group '/em/fibsem-uint16/shape'> <zarr.hierarchy.Group '/em/fibsem-uint16/dtype'>
 │       ├── s0 (8570, 3000, 10000) uint16
 │       ├── s1 (4285, 1500, 5000) uint16
 │       ├── s2 (2142, 750, 2500) uint16
 │       ├── s3 (1071, 375, 1250) uint16
 │       └── s4 (535, 187, 625) uint16
 └── labels <zarr.hierarchy.Group '/labels/shape'> <zarr.hierarchy.Group '/labels/dtype'>
     ├── cent-dapp_pred <zarr.hierarchy.Group '/labels/cent-dapp_pred/shape'> <zarr.hierarchy.Group '/labels/cent-dapp_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── cent-dapp_seg <zarr.hierarchy.Group '/labels/cent-dapp_seg/shape'> <zarr.hierarchy.Group '/labels/cent-dapp_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── cent_pred <zarr.hierarchy.Group '/labels/cent_pred/shape'> <zarr.hierarchy.Group '/labels/cent_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── cent_seg <zarr.hierarchy.Group '/labels/cent_seg/shape'> <zarr.hierarchy.Group '/labels/cent_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── chrom_pred <zarr.hierarchy.Group '/labels/chrom_pred/shape'> <zarr.hierarchy.Group '/labels/chrom_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── chrom_seg <zarr.hierarchy.Group '/labels/chrom_seg/shape'> <zarr.hierarchy.Group '/labels/chrom_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint32
     │   ├── s1 (4280, 1500, 5000) uint32
     │   ├── s2 (2140, 750, 2500) uint32
     │   ├── s3 (1070, 375, 1250) uint32
     │   └── s4 (535, 187, 625) uint32
     ├── ecs_pred <zarr.hierarchy.Group '/labels/ecs_pred/shape'> <zarr.hierarchy.Group '/labels/ecs_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── ecs_seg <zarr.hierarchy.Group '/labels/ecs_seg/shape'> <zarr.hierarchy.Group '/labels/ecs_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── endo-mem_pred <zarr.hierarchy.Group '/labels/endo-mem_pred/shape'> <zarr.hierarchy.Group '/labels/endo-mem_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── endo-mem_seg <zarr.hierarchy.Group '/labels/endo-mem_seg/shape'> <zarr.hierarchy.Group '/labels/endo-mem_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── endo_er_contacts <zarr.hierarchy.Group '/labels/endo_er_contacts/shape'> <zarr.hierarchy.Group '/labels/endo_er_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── endo_golgi_contacts <zarr.hierarchy.Group '/labels/endo_golgi_contacts/shape'> <zarr.hierarchy.Group '/labels/endo_golgi_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── endo_mt_contacts <zarr.hierarchy.Group '/labels/endo_mt_contacts/shape'> <zarr.hierarchy.Group '/labels/endo_mt_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── endo_pred <zarr.hierarchy.Group '/labels/endo_pred/shape'> <zarr.hierarchy.Group '/labels/endo_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── endo_seg <zarr.hierarchy.Group '/labels/endo_seg/shape'> <zarr.hierarchy.Group '/labels/endo_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── er-mem_pred <zarr.hierarchy.Group '/labels/er-mem_pred/shape'> <zarr.hierarchy.Group '/labels/er-mem_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── er-mem_seg <zarr.hierarchy.Group '/labels/er-mem_seg/shape'> <zarr.hierarchy.Group '/labels/er-mem_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── er_curvature <zarr.hierarchy.Group '/labels/er_curvature/shape'> <zarr.hierarchy.Group '/labels/er_curvature/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── er_golgi_contacts <zarr.hierarchy.Group '/labels/er_golgi_contacts/shape'> <zarr.hierarchy.Group '/labels/er_golgi_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── er_medial-surface <zarr.hierarchy.Group '/labels/er_medial-surface/shape'> <zarr.hierarchy.Group '/labels/er_medial-surface/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── er_mito_contacts <zarr.hierarchy.Group '/labels/er_mito_contacts/shape'> <zarr.hierarchy.Group '/labels/er_mito_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── er_mt_contacts <zarr.hierarchy.Group '/labels/er_mt_contacts/shape'> <zarr.hierarchy.Group '/labels/er_mt_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── er_pm_contacts <zarr.hierarchy.Group '/labels/er_pm_contacts/shape'> <zarr.hierarchy.Group '/labels/er_pm_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── er_pred <zarr.hierarchy.Group '/labels/er_pred/shape'> <zarr.hierarchy.Group '/labels/er_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── er_ribo_contacts <zarr.hierarchy.Group '/labels/er_ribo_contacts/shape'> <zarr.hierarchy.Group '/labels/er_ribo_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint32
     │   ├── s1 (4280, 1500, 5000) uint32
     │   ├── s2 (2140, 750, 2500) uint32
     │   ├── s3 (1070, 375, 1250) uint32
     │   └── s4 (535, 187, 625) uint32
     ├── er_seg <zarr.hierarchy.Group '/labels/er_seg/shape'> <zarr.hierarchy.Group '/labels/er_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── er_vesicle_contacts <zarr.hierarchy.Group '/labels/er_vesicle_contacts/shape'> <zarr.hierarchy.Group '/labels/er_vesicle_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── eres_pred <zarr.hierarchy.Group '/labels/eres_pred/shape'> <zarr.hierarchy.Group '/labels/eres_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── eres_seg <zarr.hierarchy.Group '/labels/eres_seg/shape'> <zarr.hierarchy.Group '/labels/eres_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── golgi-mem_pred <zarr.hierarchy.Group '/labels/golgi-mem_pred/shape'> <zarr.hierarchy.Group '/labels/golgi-mem_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── golgi-mem_seg <zarr.hierarchy.Group '/labels/golgi-mem_seg/shape'> <zarr.hierarchy.Group '/labels/golgi-mem_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── golgi_mt_contacts <zarr.hierarchy.Group '/labels/golgi_mt_contacts/shape'> <zarr.hierarchy.Group '/labels/golgi_mt_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── golgi_pred <zarr.hierarchy.Group '/labels/golgi_pred/shape'> <zarr.hierarchy.Group '/labels/golgi_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── golgi_seg <zarr.hierarchy.Group '/labels/golgi_seg/shape'> <zarr.hierarchy.Group '/labels/golgi_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── golgi_vesicle_contacts <zarr.hierarchy.Group '/labels/golgi_vesicle_contacts/shape'> <zarr.hierarchy.Group '/labels/golgi_vesicle_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── gt <zarr.hierarchy.Group '/labels/gt/shape'> <zarr.hierarchy.Group '/labels/gt/dtype'>
     │   ├── s0 (17120, 6000, 20000) uint64
     │   ├── s1 (8560, 3000, 10000) uint64
     │   ├── s2 (4280, 1500, 5000) uint64
     │   ├── s3 (2140, 750, 2500) uint64
     │   └── s4 (1070, 375, 1250) uint64
     ├── ld-mem_pred <zarr.hierarchy.Group '/labels/ld-mem_pred/shape'> <zarr.hierarchy.Group '/labels/ld-mem_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── ld-mem_seg <zarr.hierarchy.Group '/labels/ld-mem_seg/shape'> <zarr.hierarchy.Group '/labels/ld-mem_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── ld_pred <zarr.hierarchy.Group '/labels/ld_pred/shape'> <zarr.hierarchy.Group '/labels/ld_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── ld_seg <zarr.hierarchy.Group '/labels/ld_seg/shape'> <zarr.hierarchy.Group '/labels/ld_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── lyso-mem_pred <zarr.hierarchy.Group '/labels/lyso-mem_pred/shape'> <zarr.hierarchy.Group '/labels/lyso-mem_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── lyso-mem_seg <zarr.hierarchy.Group '/labels/lyso-mem_seg/shape'> <zarr.hierarchy.Group '/labels/lyso-mem_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── lyso_pred <zarr.hierarchy.Group '/labels/lyso_pred/shape'> <zarr.hierarchy.Group '/labels/lyso_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── lyso_seg <zarr.hierarchy.Group '/labels/lyso_seg/shape'> <zarr.hierarchy.Group '/labels/lyso_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── masks <zarr.hierarchy.Group '/labels/masks/shape'> <zarr.hierarchy.Group '/labels/masks/dtype'>
     │   └── foreground <zarr.hierarchy.Group '/labels/masks/foreground/shape'> <zarr.hierarchy.Group '/labels/masks/foreground/dtype'>
     │       ├── s0 (4280, 1500, 5000) uint8
     │       ├── s1 (2140, 750, 2500) uint8
     │       ├── s2 (1070, 375, 1250) uint8
     │       ├── s3 (535, 187, 625) uint8
     │       └── s4 (267, 93, 312) uint8
     ├── mito-mem_pred <zarr.hierarchy.Group '/labels/mito-mem_pred/shape'> <zarr.hierarchy.Group '/labels/mito-mem_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── mito-mem_seg <zarr.hierarchy.Group '/labels/mito-mem_seg/shape'> <zarr.hierarchy.Group '/labels/mito-mem_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── mito_mt_contacts <zarr.hierarchy.Group '/labels/mito_mt_contacts/shape'> <zarr.hierarchy.Group '/labels/mito_mt_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── mito_pm_contacts <zarr.hierarchy.Group '/labels/mito_pm_contacts/shape'> <zarr.hierarchy.Group '/labels/mito_pm_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── mito_pred <zarr.hierarchy.Group '/labels/mito_pred/shape'> <zarr.hierarchy.Group '/labels/mito_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── mito_seg <zarr.hierarchy.Group '/labels/mito_seg/shape'> <zarr.hierarchy.Group '/labels/mito_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── mt-out_seg <zarr.hierarchy.Group '/labels/mt-out_seg/shape'> <zarr.hierarchy.Group '/labels/mt-out_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── mt_nucleus_contacts <zarr.hierarchy.Group '/labels/mt_nucleus_contacts/shape'> <zarr.hierarchy.Group '/labels/mt_nucleus_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── mt_pm_contacts <zarr.hierarchy.Group '/labels/mt_pm_contacts/shape'> <zarr.hierarchy.Group '/labels/mt_pm_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── mt_vesicle_contacts <zarr.hierarchy.Group '/labels/mt_vesicle_contacts/shape'> <zarr.hierarchy.Group '/labels/mt_vesicle_contacts/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── ne-mem_pred <zarr.hierarchy.Group '/labels/ne-mem_pred/shape'> <zarr.hierarchy.Group '/labels/ne-mem_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── ne-mem_seg <zarr.hierarchy.Group '/labels/ne-mem_seg/shape'> <zarr.hierarchy.Group '/labels/ne-mem_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── ne_pred <zarr.hierarchy.Group '/labels/ne_pred/shape'> <zarr.hierarchy.Group '/labels/ne_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── ne_seg <zarr.hierarchy.Group '/labels/ne_seg/shape'> <zarr.hierarchy.Group '/labels/ne_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── nhchrom_pred <zarr.hierarchy.Group '/labels/nhchrom_pred/shape'> <zarr.hierarchy.Group '/labels/nhchrom_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── nhchrom_seg <zarr.hierarchy.Group '/labels/nhchrom_seg/shape'> <zarr.hierarchy.Group '/labels/nhchrom_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint32
     │   ├── s1 (4280, 1500, 5000) uint32
     │   ├── s2 (2140, 750, 2500) uint32
     │   ├── s3 (1070, 375, 1250) uint32
     │   └── s4 (535, 187, 625) uint32
     ├── np_pred <zarr.hierarchy.Group '/labels/np_pred/shape'> <zarr.hierarchy.Group '/labels/np_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── np_seg <zarr.hierarchy.Group '/labels/np_seg/shape'> <zarr.hierarchy.Group '/labels/np_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── nucleolus_pred <zarr.hierarchy.Group '/labels/nucleolus_pred/shape'> <zarr.hierarchy.Group '/labels/nucleolus_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── nucleolus_seg <zarr.hierarchy.Group '/labels/nucleolus_seg/shape'> <zarr.hierarchy.Group '/labels/nucleolus_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── nucleus_pred <zarr.hierarchy.Group '/labels/nucleus_pred/shape'> <zarr.hierarchy.Group '/labels/nucleus_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── nucleus_seg <zarr.hierarchy.Group '/labels/nucleus_seg/shape'> <zarr.hierarchy.Group '/labels/nucleus_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── pm_pred <zarr.hierarchy.Group '/labels/pm_pred/shape'> <zarr.hierarchy.Group '/labels/pm_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── pm_seg <zarr.hierarchy.Group '/labels/pm_seg/shape'> <zarr.hierarchy.Group '/labels/pm_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── ribo_classified <zarr.hierarchy.Group '/labels/ribo_classified/shape'> <zarr.hierarchy.Group '/labels/ribo_classified/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   └── s4 (535, 187, 625) uint8
     ├── ribo_pred <zarr.hierarchy.Group '/labels/ribo_pred/shape'> <zarr.hierarchy.Group '/labels/ribo_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── ribo_seg <zarr.hierarchy.Group '/labels/ribo_seg/shape'> <zarr.hierarchy.Group '/labels/ribo_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint32
     │   ├── s1 (4280, 1500, 5000) uint32
     │   ├── s2 (2140, 750, 2500) uint32
     │   ├── s3 (1070, 375, 1250) uint32
     │   └── s4 (535, 187, 625) uint32
     ├── vesicle-mem_pred <zarr.hierarchy.Group '/labels/vesicle-mem_pred/shape'> <zarr.hierarchy.Group '/labels/vesicle-mem_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     ├── vesicle-mem_seg <zarr.hierarchy.Group '/labels/vesicle-mem_seg/shape'> <zarr.hierarchy.Group '/labels/vesicle-mem_seg/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint16
     │   ├── s1 (4280, 1500, 5000) uint16
     │   ├── s2 (2140, 750, 2500) uint16
     │   ├── s3 (1070, 375, 1250) uint16
     │   └── s4 (535, 187, 625) uint16
     ├── vesicle_pred <zarr.hierarchy.Group '/labels/vesicle_pred/shape'> <zarr.hierarchy.Group '/labels/vesicle_pred/dtype'>
     │   ├── s0 (8560, 3000, 10000) uint8
     │   ├── s1 (4280, 1500, 5000) uint8
     │   ├── s2 (2140, 750, 2500) uint8
     │   ├── s3 (1070, 375, 1250) uint8
     │   ├── s4 (535, 188, 625) uint8
     │   └── s5 (268, 94, 313) uint8
     └── vesicle_seg <zarr.hierarchy.Group '/labels/vesicle_seg/shape'> <zarr.hierarchy.Group '/labels/vesicle_seg/dtype'>
         ├── s0 (8560, 3000, 10000) uint16
         ├── s1 (4280, 1500, 5000) uint16
         ├── s2 (2140, 750, 2500) uint16
         ├── s3 (1070, 375, 1250) uint16
         └── s4 (535, 187, 625) uint16
