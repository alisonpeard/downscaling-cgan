import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Conv2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, LeakyReLU, UpSampling2D

from blocks import residual_block, const_upscale_block


def generator(mode, # "GAN", deleted all the other options for now
              arch,
              downscaling_steps, # =[1]
              input_channels,
              constant_fields,
              filters_gen,
              latent_variables=1, # VAEGAN only
              noise_channels=4,
              conv_size=(3, 3),
              padding=None,
              relu_alpha=0.2,
              norm=None):

    forceconv = True if arch in ("forceconv", "forceconv-long") else False

    generator_input = Input(shape=(None, None, input_channels), name="condition")
    const_input = Input(shape=(None, None, constant_fields), name="constant_fields")
    upscaled_const_input = const_upscale_block(
        const_input,
        steps=downscaling_steps,
        filters=filters_gen
        )

    noise_input = Input(shape=(None, None, noise_channels), name="noise")
    generator_output = concatenate([generator_input, upscaled_const_input, noise_input])

    for ii in range(3):
        generator_output = residual_block(generator_output,
                                          filters=filters_gen,
                                          conv_size=conv_size,
                                          stride=1,
                                          relu_alpha=relu_alpha,
                                          norm=norm,
                                          padding=padding,
                                          force_1d_conv=forceconv
                                          )

    if arch == "forceconv-long":
        for ii in range(3):
            generator_output = residual_block(generator_output,
                                              filters=filters_gen,
                                              conv_size=conv_size,
                                              stride=1,
                                              relu_alpha=relu_alpha,
                                              norm=norm,
                                              padding=padding,
                                              force_1d_conv=forceconv
                                              )

    # Upsampling from low-res to high-res with alternating residual blocks
    # In the paper, this was [2*filters_gen, filters_gen] for steps of 5 and 2
    block_channels = [2*filters_gen]*(len(downscaling_steps)-1) + [filters_gen]
    for ii, step in enumerate(downscaling_steps):
        generator_output = UpSampling2D(size=(step, step), interpolation='bilinear')(generator_output)
        generator_output = residual_block(generator_output,
                                          filters=block_channels[ii],
                                          conv_size=conv_size,
                                          stride=1,
                                          relu_alpha=relu_alpha,
                                          norm=norm,
                                          padding=padding,
                                          force_1d_conv=forceconv
                                          )

    # Concatenate with original size constants field
    generator_output = concatenate([generator_output, const_input])

    for ii in range(3):
        generator_output = residual_block(generator_output,
                                          filters=filters_gen,
                                          conv_size=conv_size,
                                          stride=1,
                                          relu_alpha=relu_alpha,
                                          norm=norm,
                                          padding=padding,
                                          force_1d_conv=forceconv
                                          )

    # Output layer
    generator_output = Conv2D(filters=1, kernel_size=(1, 1), activation='softplus', name="output")(generator_output)

    model = Model(inputs=[generator_input, const_input, noise_input], outputs=generator_output, name='gen')
    return model


def discriminator(arch,
                  downscaling_steps, # =[1]
                  input_channels,
                  constant_fields,
                  filters_disc,
                  conv_size=(3, 3),
                  padding=None,      # reflection and symmetric padding options
                  stride=1,
                  relu_alpha=0.2,
                  norm=None):

    forceconv = True if arch in ("forceconv", "forceconv-long") else False

    condition = Input(shape=(None, None, input_channels), name="condition")
    constant_fields = Input(shape=(None, None, constant_fields), name="constant_fields")
    image = Input(shape=(None, None, 1), name="image")

    # convolve constant fields to match ERA5
    lo_res_constant_fields = const_upscale_block(constant_fields, steps=downscaling_steps, filters=filters_disc) # reduce resolution
    lo_res_input = concatenate([condition, lo_res_constant_fields])
    hi_res_input = concatenate([image, constant_fields])

    # encode inputs using residual blocks, in the paper, this was [filters_disc, 2*filters_disc] for steps of 5 and 2
    block_channels = [filters_disc]*(len(downscaling_steps)-1) + [2*filters_disc]
    for ii, step in enumerate(downscaling_steps):
        lo_res_input = residual_block(lo_res_input, filters=block_channels[ii], conv_size=conv_size, stride=1, relu_alpha=relu_alpha, norm=norm, padding=padding, force_1d_conv=forceconv)
        hi_res_input = Conv2D(filters=block_channels[ii], kernel_size=(step, step), strides=step, padding="valid", activation="relu")(hi_res_input)
        hi_res_input = residual_block(hi_res_input, filters=block_channels[ii], conv_size=conv_size, stride=1, relu_alpha=relu_alpha, norm=norm, padding=padding, force_1d_conv=forceconv)

    # concatenate hi- and lo-res inputs channel-wise before passing through discriminator
    disc_input = concatenate([lo_res_input, hi_res_input])

    # encode in residual blocks
    disc_input = residual_block(disc_input, filters=filters_disc, conv_size=conv_size, stride=1, relu_alpha=relu_alpha, norm=norm, padding=padding, force_1d_conv=forceconv)
    disc_input = residual_block(disc_input, filters=filters_disc, conv_size=conv_size, stride=1, relu_alpha=relu_alpha, norm=norm, padding=padding, force_1d_conv=forceconv)
    disc_input = residual_block(disc_input, filters=filters_disc, conv_size=conv_size, stride=1, relu_alpha=relu_alpha, norm=norm, padding=padding, force_1d_conv=forceconv)

    # discriminator output
    disc_output = GlobalAveragePooling2D()(disc_input)
    disc_output = Dense(64, activation='relu')(disc_output)
    disc_output = Dense(1, name="disc_output")(disc_output)

    # define model and output
    disc = Model(inputs=[condition, constant_fields, image], outputs=disc_output, name='disc')
    return disc
