import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

img_height = img_width = 32


ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'images/training_set/',
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    # batch_size=None,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training",
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'images/training_set/',
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    # batch_size=None,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation",
)

ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    'images/testing_set/',
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    # batch_size=None,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    # validation_split=0.1,
    # subset="validation",
)


# datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     horizontal_flip=False,
#     vertical_flip=False,
#     data_format='channels_last',
#     validation_split=0.0,
#     dtype=tf.float32
# )
# train_generator = datagen.flow_from_directory(
#     directory=r"./images/training_set/",
#     target_size=(img_height, img_width),
#     color_mode="rgb",
#     # batch_size=5,
#     class_mode="categorical",
#     shuffle=True,
#     seed=4
# )
#
# valid_generator = datagen.flow_from_directory(
#     directory=r"./images/valid_set/",
#     target_size=(img_height, img_width),
#     color_mode="rgb",
#     # batch_size=1,
#     class_mode="categorical",
#     shuffle=True,
#     seed=4
# )

# data = keras.datasets.cifar100
# (train_images, train_labels), (test_images, test_labels) = data.load_data()

# train_images = train_images/255.0
# test_images = test_images/255.0



def augment(image, label):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
    return image, label


ds_train = ds_train.map(augment)

# url = 'https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5'
# base_model = hub.KerasLayer(url, input_shape=(None, img_height, img_width, 3))
# base_model.trainable = False

model = keras.Sequential(
    [
        keras.Input(shape=(img_height, img_width, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2)),
        # layers.Conv2D(128, 3, activation='relu'),
        # layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(64),
        layers.Dense(32),
        layers.Dense(16),
        # base_model,
        # layers.Dense(128, activation='relu'),
        layers.Dense(2, activation="softmax"),
    ]
)

print(model.summary())

# import sys
# sys.exit()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss=[keras.losses.sparse_categorical_crossentropy,
          ],
    metrics=["accuracy"]
)

# STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
# STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
model.fit(ds_train,
          # steps_per_epoch=STEP_SIZE_TRAIN,
          validation_data=ds_validation,
          # validation_steps=STEP_SIZE_VALID,
          epochs=24)

test_loss, test_acc = model.evaluate(ds_test)

print("Tested Acc:", test_acc)

# model.save('saved_model/')
# model1 = keras.models.load_model('saved_model/')
# print(model1.summary())
# plt.imshow(train_images[0], cmap=plt.cm.binary)
# plt.show()
