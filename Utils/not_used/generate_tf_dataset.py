import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# reference https://www.youtube.com/watch?v=VFEOskzhhbc


def get_label(filepath):
    return tf.strings.split(filepath, os.path.sep)[-2]


def process_image(filepath):
    tf_label = get_label(filepath)
    tf_image = tf.io.read_file(filepath)
    tf_image = tf.image.decode_jpeg(tf_image)
    tf_image = tf.image.resize(tf_image, [128, 128])
    tf_image = tf_image/255.0   # scale down

    return tf_image, tf_label


images_ds = tf.data.Dataset.list_files('experimental/images/training_set/*/*', shuffle=True)
# class_names = ["true", "false"]

image_cnt = len(images_ds)

train_size = int(image_cnt*0.8)

train_ds = images_ds.take(train_size)   # take 80% of image for training
test_ds = images_ds.skip(train_size)    # take 20% of image for testing

train_ds = train_ds.map(process_image)
test_ds = test_ds.map(process_image)


for image, label in train_ds:  # debug
    print("label: ", label.numpy())
    print("image: ", image.numpy()[0][0])

# for image, label in test_ds:
#     imgpy = image.numpy()
#     plt.imshow(imgpy)
#     plt.show()











