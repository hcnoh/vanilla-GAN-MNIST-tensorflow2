import os
import pickle

import numpy as np
import tensorflow as tf

import config as conf

from data.mnist import Mnist
from models.vanilla_gan import VanillaGAN


def feature_normalize(features):
    return (features - 0.5) / 0.5


def feature_denormalize(features):
    return (features + 1) / 2


def main():
    model_spec_name = "%s-model-spec.json" % conf.MODEL_NAME
    model_rslt_name = "%s-results.pickle" % conf.MODEL_NAME

    model_save_path = os.path.join(conf.MODEL_SAVE_DIR, conf.MODEL_NAME)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    model_ckpt_path = os.path.join(model_save_path, "model-ckpt")
    model_spec_path = os.path.join(model_save_path, model_spec_name)
    model_rslt_path = os.path.join(model_save_path, model_rslt_name)

    loader = Mnist()

    features = np.vstack([loader.train_features, loader.test_features])
    features = feature_normalize(features).astype(np.float32)

    num_sets = loader.num_train_sets + loader.num_test_sets
    
    feature_depth = loader.feature_depth
    feature_shape = loader.feature_shape

    latent_depth = conf.LATENT_DEPTH

    batch_size = conf.BATCH_SIZE
    num_epochs = conf.NUM_EPOCHS

    model = VanillaGAN(feature_depth)

    generator_opt = tf.keras.optimizers.Adam(learning_rate=0.0002)
    discriminator_opt = tf.keras.optimizers.Adam(learning_rate=0.0002)

    @tf.function
    def train_step(x, z):
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generator_loss = model.generator_loss(z)
            discriminator_loss = model.discriminator_loss(x, z)

            grads_generator_loss = generator_tape.gradient(
                target=generator_loss, sources=model.generator.trainable_variables
            )
            grads_discriminator_loss = discriminator_tape.gradient(
                target=discriminator_loss, sources=model.discriminator.trainable_variables
            )

            discriminator_opt.apply_gradients(
                zip(grads_discriminator_loss, model.discriminator.trainable_variables)
            )
            generator_opt.apply_gradients(
                zip(grads_generator_loss, model.generator.trainable_variables)
            )

        return generator_loss, discriminator_loss

    ckpt = tf.train.Checkpoint(generator=model.generator, discriminator=model.discriminator)

    steps_per_epoch = num_sets // batch_size
    train_steps = steps_per_epoch * num_epochs

    generator_losses = []
    discriminator_losses = []
    generator_losses_epoch = []
    discriminator_losses_epoch = []
    x_fakes = []
    for i in range(1, train_steps+1):
        epoch = i // steps_per_epoch

        idxes = np.random.choice(num_sets, batch_size, replace=False)
        x_i = features[idxes]
        z_i = np.random.normal(size=[batch_size, latent_depth]).astype(np.float32)

        generator_loss_i, discriminator_loss_i = train_step(x_i, z_i)
        
        generator_losses.append(generator_loss_i)
        discriminator_losses.append(discriminator_loss_i)

        if i % steps_per_epoch == 0:
            x_fake = model.generator(z_i, training=False)
            x_fake = feature_denormalize(x_fake)

            generator_loss_epoch = np.mean(generator_losses[-steps_per_epoch:])
            discriminator_loss_epoch = np.mean(discriminator_losses[-steps_per_epoch:])

            print("Epoch: %i,  Generator Loss: %f,  Discriminator Loss: %f" % \
                (epoch, generator_loss_epoch, discriminator_loss_epoch)
            )

            generator_losses_epoch.append(generator_loss_epoch)
            discriminator_losses_epoch.append(discriminator_loss_epoch)

            x_fakes.append(x_fake)
            
            ckpt.save(file_prefix=model_ckpt_path)

            with open(model_rslt_path, "wb") as f:
                pickle.dump((generator_losses_epoch, discriminator_losses_epoch, x_fakes), f)


if __name__ == "__main__":
    main()