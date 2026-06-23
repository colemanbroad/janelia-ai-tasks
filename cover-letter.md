# Prompt

Describe a deep learning project you have executed—ideally a creative use of a vision
transformer, U-Net architecture, or Diffusion model that you trained yourself. Projects in
computer vision for microscopy image analysis are especially relevant.

Include a link to a code repository if possible. 

If you contributed to a joint project, please describe your specific contributions. 

Briefly discuss the project's results, limitations, and challenges you encountered. 

# Letter

I designed a self-supervised denoising model that generalizes the Noise2Void approach
by decoupling the sets of randomized (masked) pixels in the input from those in the loss.
This enables the removal of structured noise given knowledge of the support of its autocorrelation function.
The design is largely independent of architecture, but we used a U-net in the experiments.
Our approach significantly outperformed standard denoisers including Noise2Void and Non-local means on
bioimage datasets with structured noise including images of fluorescent cell membranes acquired by spinning disk microscopy,
3D images of C. elegans nuclei and Zebrafish retina (scanning confocal).

When compared with Noise2Void this approach requires knowledge of the shape of the noise mask, which can be computed from a pure-noise region of the image,
but otherwise is a significant unknown hyperparameter.
Using a convolutional architecture requires that the support of the noise correlation be constant across the image.

The origins of the structured noise were often unknown to us.
On the one hand it is great that the method doesn't require this knowledge,
but it's limitation that we couldn't compare against source-specific denoisig techniques.

The code is available [here](https://github.com/mpicbg-csbd/structured_N2V).

