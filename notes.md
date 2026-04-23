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

FIX: The DINOv3 download links don't work with wget or curl! It's a mystery.
Let's start off with the smallest model 21M.

OK, the RGB plot of top 3 PCA dims reveal a very weak correlation with image content and a
very strong correlation with spatial position! So either
1. there's something wrong with our rope embeddings
2. there's something wrong with our image input norm
3. the image resolution is so wrong that the model is producing random noise
4. the 21M model doesn't have the features this data needs

Let's do the same RGB-PCA plot for a few samples of noise...

Indeed, there's very strong correlation with the same results for an image of noise.

Let's try a higher resolution and see what changes...

It looks a little better but not much. The model is able to identify large light and dark regions, but still strong correlations.

Let's try normalizing according to ImgNet standards...

Yes! This removed the spatial rainbow and enhanced the content.

Let's try this on the high res image...

OK, at s2 resolution it picks up large scale features the size of tens of mitos, but most
of the mito areas are a sea of blue-green without obvious pattern. At higher resolutions the
results look quite noisy. I need a different view to see if the mitos are being identified.
Maybe we can cluster the dense predictions into a small number of types and plot the image
patches tiled by cluster?

Let's see if we can directly change the patch_embed.proj.stride to made predictions more dense...
Yes, this doesn't break the structure of the model. The stride is flexible.

--- ---

# Questions

Q: How are we going to validate clusters without supervised GT?
A: Some datasets DO have GT, but this is out of scope. Let's just eyeball it.

## Task 2.2.1 -- Patch Size Selection

The pretrained models require patches of size 16x16.
Since mitos are typically 20px across at their most narrow in the s2 data this embedding doesn't
capture their full structure. So it's unlikely to capture mito geometry, but will get texture.
We can avg-pool/downscale the images until the average mito fits inside the 16x16 window.
We could even use the small amount of ground truth segmentations to determine the scaling ratio.
Or we can treat the image scale as a hyperparam to be fit/trained against ground truth segmentations.

In the end we just tried s0/s1/s2/s3 and eyeballed the PCA image embeddings.
s2 appeared most effective for mouse liver data as it had the strongest visual correlation with mitos.

## Task 2.2.2 -- Even more dense embeddings

The DINO paper refers to per-patch embeddings that tile a full image as "dense",
but sense we're interested in detailed segmentations of small objects we need per-pixel embeddings. 
To increase the resolution of our predictions we can
0. take the stride-16 embeddings and upsample them with e.g. bilinear interpolation (this is what the paper does).
1. reduce the model's patch-embedding stride to create overlapping embeddings and then average the results.
2. equivalently, we can apply `avg(Tinv(model(T(x))))` for whole-image translations `T`.
3. we can extend `T` to be any kind of information-preserving transformation over which our embeddings should be invariant.
4. we can train a super-resolution model to intelligently enhance the results.


---

It's probably fair game to ask how these DINO embeddings compare with classical hand-coded feature extractors, e.g. SIFT.
At stride=8 it's hard to notice any visual patterns for the mitos, but at stride=4 they emerge and at stride=2 you can count them.
The LBP features reveal some global patterns but don't really pick up on mitos.

---

Let's try this approach with the s0 and s1 resolution data.
The s1 res data with stride=4 DINO still allows you to see/count mitos, but the color signal is
significantly weaker. The red mitos don't stand out as well against the blue background.

For completeness we'll try the s0 image...

The stride=16 version is waaaay too noisy. Very little RGB correlation with mito.
The stride=8 version is still way too noisy. R correlates with dark pixels, G with edges. B is everywhere.

Exploring hyperopt for LBP (f9) shows r4p32 to have the most interesting correlation with mitos.

## Task 2.3.1 -- Embedding-Based Retrieval & Visualization

Pick out a test mito and use it as a query to evaluate quality of embeddings wrt mitos specifically.
There isn't an unambiguous 1-1 mapping between embeddings and mitos, but we can
1. average an embedding over the mito mask
2. pick the embedding closest to the mito centerpoint
3. and also choose a representative mito
4. or choose a mito that fits in the 16x16 patch size

Then instead of doing PCA on the embeddings we can take the cosine similarity between
our query mito and the rest of the image. The result will be a greyscale image showing similarity
to the query mito. If we're using multiple query objects then we can use the average/max cosine
similarity.

Let's start off with the query point (107, 317) in the s2 data...

This correctly identifies many mitos, but there are also many false positives in bright areas.
TODO: try computing not just the similarity to a query mito, but the *dissimilarity* to a chosen
bright background point like (232, 230)?

This doesn't work at all. It just reveals dark gruanular patches.

Ok, let's see how well this query point works on a different dataset...

There is a mito centerpoint at (165,250) in the mouse kidney.
These mitos are very dark, and a simple blur+threshold could be enough to segment them.



##  Task 2.4.1 -- Improving on DINO

Many possible approaches.
We can train a Vision Transformer (ViT) from scratch on EM data with self-supervised DINO-style loss, but that involves many trainable params.
We could even train a new ViT per EM dataset to avoid domain shift (certainly unnecessary).
Alternatively, we could fine-tune an existing DINO model using e.g. Low rank adaptation (LoRA)
in the same way, although typically LoRA works well for deep, abstract features and is less
useful for initial, shallow features.
However, I expect this NOT to be the case here, because EM data is fundamentally different
from the natural images on which DINO was trained.
TODO: try using a model trained on satellite imagery. It may be a better domain match.
We could also do linear-probing using a small amount of the existing ground truth. 
