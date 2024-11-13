# Notes
## General questions
1. Why do convolutional blocks have reflect/symmetric padding? Did this make a difference?
2. Is there much difference between the `arch` options, e.g., `forceconv`, ...
3. Why is the activation first in the residual blocks?
8. The noise has the image shape + 4 channels. Is this to match the number of classes? This is pretty big for a GAN, was this found to make a difference? 
9. Also, can't find intermediate regularizing noise steps, maybe misunderstood.
6. Why is the specific non-trainable environment needed/why create training networks? Is it just a style choice?
7. Why are there multiple losses and how do they interact? There are three losses for the generator, two for the critic's real and fake labels (`wassertstein_loss`) and one for the gradient-penalty. It seems like instead of minimising the distance between $D(G(z))$ and $D(x)$, we specifically want them to approach $\{-1, 1\}$.
10. Is `_parse_batch()` just a more complete implementation of `__str__`
12. Why is garbage collection used?
13. What format is original data in?


## Data structures
* Forecast fields: `['cape', 'cp', 'mcc', 'sp', 'ssr', 't2m', 'tciw', 'tclw', 'tcrw', 'tcw', 'tcwv', 'tp', 'u700', 'v700']`
* Forecast data (condition): `FCST_PATH/<yearstr>/<field>.nc`
* Constant fields: `CONSTANTS_PATH/elev.nc` (oro), `CONSTANTS_PATH/lsm.nc` (land-sea mask)
* Truth data: `TRUTH_PATH/<date>_<hr>.nc4`
* Forecast data is expected to have _mean and _sd fields, I will set sd=0
* Can just use (I think) truth data with `autocoarsen` setting, not currently

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
* Creates a non-trainable container

## Main / configuration
* `forceconv-long` 6 residual blocks instead of 3, option from `arch`
* `latent_variables=50` per pixel -- VAEGAN
* `noise_channels=4` -- GAN
    * `Input(shape=(None, None, noise_channels), name="noise")`
    * Concatenate this with condition and constant fields then pass through model
    * Use a noise generator in `train.py` and pass generator into `train()`
         ```
        noise_shape = (img_shape[0], img_shape[1], noise_channels)
        noise_gen = noise.NoiseGenerator(noise_shape, batch_size=batch_size)
        ```
    * I just sample as part of the GAN without a separate class, think that's okay too
    * Main point is the latent space is way bigger, it's the same size as the image and has ~4 channels. Are the four channels for the four classses?
* What is `gc.collect()`? Garbage collection. What benefit does this provide?
* uses batchgenerator from `tfrecords`


## Data handling / batching
* If I use `autocoarsen` it will just automatically coarsen my dataset and use that for training
* `gen_fcst_norm` generates and saves the image dataset stats to use in training
* `tfrecords_generator.py`: subsamples full size images and saves as tfrecord datasets (for efficiency?)
    * Create a list of datasets for each class using `create_dataset()`
        * Loads files from `"<year>_*.<class>.tfrecords"`in TF Records dir
        * Builds a `TFRecordDataset` with `tf.data.AUTOTUNE`
        * Does `_parse_batch()`:
            * Uses `tf.io.FixedLenFeature` unclear for what
            * I think all this is just a more complete version of `__str__()`?
        * Does `ds.repeat()` if `repeat is True` which allows infinite generation
    * Use `tf.data.Dataset.sample_from_datasets(datasets:list, weights:list).batch(batch_size)`
    * This passes through a few wrappers but is basically passed into `train.train_model()` unchanged
* `data_generator.py`:
    * ...

