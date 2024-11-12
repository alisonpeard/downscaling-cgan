# Notes
## General features
1. Convolutional blocks with reflect/symmetric padding, why?
2. What is `arch` argument? `['forceconv', 'forceconv-long']`
3. Why is the activation first in the residual blocks?
4. Can't find the extra noise layers mentioned.
5. Are ensembles used for both WGAN-GP and VAEGAN or just VAEGAN? What is the content loss here?
6. Why is the specific non-trainable environment needed/why create training networks? Is it just a style choice?
7. What is the purpose of the three `loss_weights` when compiling the generator in `gan.py`? Why are there multiple losses and how do they interact? There are three losses for the generator, two for the critic's real and fake labels (`wassertstein_loss`) and one for the gradient-penalty. It seems like instead of minimising the distance between $D(G(z))$ and $D(x)$, we specifically want them to approach $\{-1, 1\}$.
8. New latent noise and batch sample for each discriminator iter
    * My code reuses the same latent space for all critic iters I should change this
    * Check what Gulrajani (2017) did

## Discriminator features
* Location: `models.py`
* Input: 
    * High-resolution constant inputs, e.g., elevation
    * Condition `(None, None, nchannels)`
    * Image samples
* Steps:
    * Concatenate constant fields to both condition (low res) and image
    * For n steps apply residual blocks to both and downscaling (reduce res) to image + constant_fields
    * Concatenate high and low res inputs
    * Three more residual blocks
    * Global average pooling + two dense layers
* Training (from `layers.py`):
    * `RandomWeightedAverage()` same as interpolation in my code
    * `GradientPenalty()` class:
        same as mine
        $$\sqrt{\sum \nabla D(\bar{x})} - 1$$


## Generator features
* Location: `models.py`
* Input:
    * Condition `(None, None, nchannels)` 
    * High-resolution constant inputs, e.g., elevation
    * Random latent vector `shape=(None, None, noise_channels)`
* Steps:
    * Reduce resolution of constant fields
    * Concatenate all three inputs
    * Three residual blocks (reflect padding)
    * Three more residual blocks if `arch == "forceconv-long"`
    * Alternate upsampling and residual blocks for n downscaling steps --> high resolution
    * Concatenate high-res generated image with constant fields again
    * Three more residual blocks (reflect padding)
    * Output: Conv2D with 1 filter and 1 kernal size and softplus
* Training:
    * Create a training network
    * What are the loss weights about?


## Residual block features
* Location: `blocks.py`
* Steps:
    * Adjust `x_in` to have right height, width, and depth
    * Block 1:
        * Leaky ReLU
        * 3x3 Conv2D (with reflect padding, option for strided convolution)
        * (Optional) batch normalization
    * Block 2:
        * Leaky ReLU
        * 3x3 Conv2D (with reflect padding, no stride)
        * (Optional) batch normalization
    * Add `x` + `x_in`


## Nontrainable class
* Location: `meta.py`
* Creates a non-trainable container to use with `with`

