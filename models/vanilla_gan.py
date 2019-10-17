import tensorflow as tf


def generator(feature_depth, name, units_list=[128, 256, 256]):
    model = tf.keras.Sequential(name=name)
    for i, units in enumerate(units_list):
        model.add(tf.keras.layers.Dense(units=units, activation=tf.nn.leaky_relu, name="dense_%i" % i))
    model.add(tf.keras.layers.Dense(units=feature_depth, activation=tf.nn.tanh, name="dense_final"))

    return model


def discriminator(name, units_list=[256, 256, 128]):
    model = tf.keras.Sequential(name=name)
    for i, units in enumerate(units_list):
        model.add(tf.keras.layers.Dense(units=units, activation=tf.nn.relu, name="dense_%i" % i))
    model.add(tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid, name="dense_final"))

    return model


class VanillaGAN(object):
    def __init__(self, feature_depth):
        self.feature_depth = feature_depth

        self.generator = generator(self.feature_depth, "generator")
        self.discriminator = discriminator("discriminator")
    
    def generator_loss(self, z):
        x = self.generator(z, training=True)
        score = self.discriminator(x, training=False)

        loss = tf.keras.losses.binary_crossentropy(
            y_true=tf.ones_like(score), y_pred=score, from_logits=False
        )

        return loss
    
    def discriminator_loss(self, x, z):
        x_fake = self.generator(z, training=False)
        true_score = self.discriminator(x, training=True)
        fake_score = self.discriminator(x_fake, training=True)

        loss = \
            tf.keras.losses.binary_crossentropy(
                y_true=tf.ones_like(true_score), y_pred=true_score, from_logits=False
            ) + \
            tf.keras.losses.binary_crossentropy(
                y_true=tf.zeros_like(fake_score), y_pred=fake_score, from_logits=False
            )
        
        return loss