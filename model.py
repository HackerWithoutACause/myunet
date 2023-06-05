import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

import os
import sys
from tqdm import tqdm

DIMENSIONS = (128, 128, 3)
inputs = layers.Input(DIMENSIONS)
normalized = layers.Lambda(lambda x: x / 255)(inputs)

def convolution(filters, index, place):
    return layers.Conv2D(
        filters,
        (3, 3),
        activation='relu',
        name='Conv2D-%s-%d' % (place, index),
        kernel_initializer='he_normal',
        padding='same'
    )

class UNetStage(keras.layers.Layer):
    def __init__(self, filters, index):
        super().__init__()
        self.c1 = convolution(filters, index, 'first')
        self.dropout = layers.Dropout(0.1)
        self.c2 = convolution(filters, index, 'second')

    def call(self, inputs):
        return self.c2(self.dropout(self.c1(inputs)))

def expansive(stright, cross, filters, index):
    transposed = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(stright)
    cat = layers.concatenate([transposed, cross])
    cv1 = convolution(filters, index, 'expansive-first')(cat)
    drop = layers.Dropout(0.2, name='Dropout-%d' % index)(cv1)
    return convolution(filters, index, 'expansive-second')(drop)

stages = [UNetStage(16, 0)(normalized)]
for i, filters in enumerate(range(5, 9)):
    pooling = layers.MaxPooling2D(pool_size=(2, 2))(stages[-1])
    stages.append(UNetStage(2 ** filters, i + 1)(pooling))

reversed = iter(reversed(stages))
last = next(reversed)

for i, con in enumerate(reversed):
    last = expansive(last, con, 2 ** (-i + 7), i)

outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(last)

model = keras.Model(inputs=[inputs], outputs=[outputs], name='unet')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def load_image_and_mask(image_path, mask_path):
    img = imread(image_path)
    img = resize(img, DIMENSIONS, mode='constant', preserve_range=True)
    mask = imread(mask_path)
    mask = resize(mask, (*DIMENSIONS[:2], 1), mode='constant', preserve_range=True)
    return (img, mask)

if len(sys.argv) < 2:
    print('Error: Add argument of train or run')
elif sys.argv[1] == 'train':
    images_path = sys.argv[2]
    mask_path = sys.argv[3]

    image_names = [name.split('.')[0] for name in next(os.walk(images_path))[2]]

    images = np.zeros((len(image_names), *DIMENSIONS), dtype=np.uint8)
    masks = np.zeros((len(image_names), *(*DIMENSIONS[:2], 1)), dtype=bool)

    for n, id in tqdm(enumerate(image_names), total=len(image_names)):
        (images[n], masks[n]) = load_image_and_mask(images_path + id + '.png', mask_path + id + '_seg0.png')

    callbacks = [
        keras.callbacks.ModelCheckpoint('model.h5', verbose=1, save_best_only=True),
        keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        keras.callbacks.TensorBoard(log_dir='logs'),
    ]
    results = model.fit(images, masks, validation_split=0.1, batch_size=15, epochs=25, callbacks=callbacks)
elif sys.argv[1] == 'run':
    model.load_weights('model.h5')
    img = imread(sys.argv[2])
    img = resize(img, DIMENSIONS[:2], mode='constant', preserve_range=True)[:, :, :3]
    imsave('input.png', img.astype(np.uint8))
    img = np.expand_dims(img, axis=0)
    mask = model(img, training=False)
    threshold = 0.5
    mask = np.where(mask >= threshold, 1, 0)
    imsave('output.png', np.squeeze(mask * 255).astype(np.uint8))
elif sys.argv[1] == 'summary':
    model.summary()
else:
    print('Error: Add argument of train or run')