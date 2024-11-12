import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.python.keras.utils import generic_utils

from layers import GradientPenalty, RandomWeightedAverage
from meta import ensure_list, input_shapes, Nontrainable, load_opt_weights, save_opt_weights
# from vaegantrain import VAE_trainer
from wloss import wasserstein_loss, CL_chooser


class WGANGP(object):

    def __init__(self, gen, disc, mode, gradient_penalty_weight=10,
                 lr_disc=0.0001, lr_gen=0.0001, avg_seed=None,
                 kl_weight=None, ensemble_size=None, CLtype=None,
                 content_loss_weight=None):

        self.gen = gen
        self.disc = disc
        self.mode = "GAN"
        self.gradient_penalty_weight = gradient_penalty_weight
        self.learning_rate_disc = lr_disc
        self.learning_rate_gen = lr_gen
        self.kl_weight = kl_weight
        self.ensemble_size = ensemble_size
        self.CLtype = CLtype
        self.content_loss_weight = content_loss_weight
        self.build_wgan_gp()

    def filenames_from_root(self, root):
        fn = {
            "gen_weights": root+"-gen_weights.h5",
            "disc_weights": root+"-disc_weights.h5",
            "gen_opt_weights": root+"-gen_opt_weights.h5",
            "disc_opt_weights": root+"-disc_opt_weights.h5"
        }
        return fn

    def load(self, load_files):
        self.gen.load_weights(load_files["gen_weights"])
        self.disc.load_weights(load_files["disc_weights"])

        with Nontrainable(self.disc):
            self.gen_trainer.make_train_function()
            load_opt_weights(self.gen_trainer,
                             load_files["gen_opt_weights"])
        with Nontrainable(self.gen):
            self.disc_trainer.make_train_function()
            load_opt_weights(self.disc_trainer,
                             load_files["disc_opt_weights"])

    def save(self, save_fn_root):
        paths = self.filenames_from_root(save_fn_root)
        self.gen.save_weights(paths["gen_weights"], overwrite=True)
        self.disc.save_weights(paths["disc_weights"], overwrite=True)
        save_opt_weights(self.disc_trainer, paths["disc_opt_weights"])
        save_opt_weights(self.gen_trainer, paths["gen_opt_weights"])

    def build_wgan_gp(self):
        # find shapes for inputs
        cond_shapes = input_shapes(self.gen, "condition")
        const_shapes = input_shapes(self.gen, "constant_fields")
        noise_shapes = input_shapes(self.gen, "noise")
        sample_shapes = input_shapes(self.disc, "image")

        # Create generator training network
        with Nontrainable(self.disc):
            cond_in = [Input(shape=cond_shapes[0])]
            const_in = [Input(shape=const_shapes[0])]

            if self.ensemble_size is None:
                noise_in = [Input(shape=noise_shapes[0])]
            else:
                noise_in = [Input(shape=noise_shapes[0])
                            for _ in range(self.ensemble_size + 1)]
            gen_in = cond_in + const_in + noise_in

            gen_out = self.gen(gen_in[0:3])  # only use cond/const/noise[0]
            gen_out = ensure_list(gen_out)
            disc_in_gen = cond_in + const_in + gen_out
            disc_out_gen = self.disc(disc_in_gen)
            full_gen_out = [disc_out_gen]
            self.gen_trainer = Model(inputs=gen_in,
                                        outputs=full_gen_out,
                                        name='gen_trainer')

        # Create discriminator training network
        with Nontrainable(self.gen):
            cond_in = [Input(shape=s, name='condition') for s in cond_shapes]
            const_in = [Input(shape=s, name='constant_fields') for s in const_shapes]
            noise_in = [Input(shape=s, name='noise') for s in noise_shapes]
            sample_in = [Input(shape=s, name='image') for s in sample_shapes]
            gen_in = cond_in + const_in + noise_in
            disc_in_real = sample_in[0]
            disc_in_fake = self.gen(gen_in)
            disc_in_avg = RandomWeightedAverage()([disc_in_real, disc_in_fake])
            disc_out_real = self.disc(cond_in + const_in + [disc_in_real])
            disc_out_fake = self.disc(cond_in + const_in + [disc_in_fake])
            disc_out_avg = self.disc(cond_in + const_in + [disc_in_avg])
            disc_gp = GradientPenalty()([disc_out_avg, disc_in_avg])
            self.disc_trainer = Model(inputs=cond_in + const_in + noise_in + sample_in,
                                      outputs=[disc_out_real, disc_out_fake, disc_gp],
                                      name='disc_trainer')

        self.compile()


    def compile(self, opt_disc=None, opt_gen=None):
        # create optimizers
        if opt_disc is None:
            opt_disc = Adam(learning_rate=self.learning_rate_disc, beta_1=0.0, beta_2=0.999)
        self.opt_disc = opt_disc
        if opt_gen is None:
            opt_gen = Adam(learning_rate=self.learning_rate_gen, beta_1=0.0, beta_2=0.999)
        self.opt_gen = opt_gen

        with Nontrainable(self.disc):
            if self.ensemble_size is not None:
                CLfn = CL_chooser(self.CLtype)
                losses = [wasserstein_loss, CLfn]
                loss_weights = [1.0, self.content_loss_weight]
            else:
                losses = [wasserstein_loss]
                loss_weights = [1.0]
            self.gen_trainer.compile(loss=losses,
                                        loss_weights=loss_weights,
                                        optimizer=self.opt_gen)
        with Nontrainable(self.gen):
            self.disc_trainer.compile(
                loss=[wasserstein_loss, wasserstein_loss, 'mse'],
                loss_weights=[1.0, 1.0, self.gradient_penalty_weight],
                optimizer=self.opt_disc
            )

    def train(self, batch_gen, noise_gen, num_gen_batches=1,
              training_ratio=1, show_progress=True):

        disc_target_real = None
        for inputs, _ in batch_gen.take(1).as_numpy_iterator():
            tmp_batch = inputs["condition"]
            batch_size = tmp_batch.shape[0]
        del tmp_batch
        del inputs
        if show_progress:
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(num_gen_batches*batch_size)
        disc_target_real = np.ones((batch_size, 1), dtype=np.float32)
        disc_target_fake = -disc_target_real
        gen_target = disc_target_real
        target_gp = np.zeros((batch_size, 1), dtype=np.float32)
        disc_target = [disc_target_real, disc_target_fake, target_gp]

        batch_gen_iter = iter(batch_gen)

        for _ in range(num_gen_batches):

            # train discriminator
            disc_loss = None
            disc_loss_n = 0
            for _ in range(training_ratio):
                # generate some real samples
                inputs, outputs = batch_gen_iter.get_next()
                cond = inputs["condition"]
                const = inputs["constant_fields"]
                sample = outputs["image"]

                with Nontrainable(self.gen):
                    dl = self.disc_trainer.train_on_batch(
                        [cond, const, noise_gen(), sample], disc_target)

                if disc_loss is None:
                    disc_loss = np.array(dl)
                else:
                    disc_loss += np.array(dl)
                disc_loss_n += 1

                del sample, cond, const

            disc_loss /= disc_loss_n

            with Nontrainable(self.disc):
                inputs, outputs = batch_gen_iter.get_next()
                cond = inputs["condition"]
                const = inputs["constant_fields"]
                sample = outputs["image"]

                condconst = [cond, const]
                if self.ensemble_size is None:
                    gt_outputs = [gen_target]
                    noise_list = [noise_gen()]
                else:
                    noise_list = [noise_gen()
                                  for ii in range(self.ensemble_size + 1)]
                    gt_outputs = [gen_target, sample]
                gt_inputs = condconst + noise_list

                gen_loss = self.gen_trainer.train_on_batch(
                    gt_inputs, gt_outputs)

                gen_loss = ensure_list(gen_loss)
                del sample, cond, const

            if show_progress:
                losses = []
                for ii, dl in enumerate(disc_loss):
                    losses.append((f"D{ii}", dl))
                for ii, gl in enumerate(gen_loss):
                    losses.append((f"G{ii}", gl))
                progbar.add(batch_size,
                            values=losses)

            loss_log = {}
            loss_log["disc_loss"] = disc_loss[0]
            loss_log["disc_loss_real"] = disc_loss[1]
            loss_log["disc_loss_fake"] = disc_loss[2]
            loss_log["disc_loss_gp"] = disc_loss[3]
            loss_log["gen_loss_total"] = gen_loss[0]
            if self.ensemble_size is not None:
                loss_log["gen_loss_disc"] = gen_loss[1]
                loss_log["gen_loss_ct"] = gen_loss[2]
            gc.collect()

        return loss_log
