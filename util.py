import tensorflow as tf


def image_summaries(name, img, max_outputs=1):
    img = tf.unstack(tf.transpose(img, [0, 2, 3, 1]), axis=-1)
    img = tf.stack([img[2], img[1], img[0]], axis=-1, name=name)
    tf.summary.image(name, img, max_outputs=max_outputs)
