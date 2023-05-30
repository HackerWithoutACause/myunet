import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DIMENSIONS = (128, 128, 3)
inputs = layers.Input(DIMENSIONS)
normalized = layers.Lambda(lambda x: x / 255)(inputs)

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result

def convolution(filters, index, place):
    return layers.Conv2D(
        filters,
        (3, 3),
        activation='relu',
        name='Conv2D-%s-%d' % (place, index),
        kernel_initializer='he_normal',
        padding='same'
    )

def stage(filters, index):
    return keras.Sequential(
        [
            convolution(filters, index, 'first'),
            layers.Dropout(0.1, name='Dropout=%d' % index),
            convolution(filters, index, 'second'),
        ]
    )

def expansive(stright, cross, filters, index):
    transposed = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(stright)
    cat = layers.concatenate([transposed, cross])
    cv1 = convolution(filters, index, 'expansive-second')(cat)
    drop = layers.Dropout(0.2)(cv1)
    cv2 = convolution(filters, index, 'expansive-second')(drop)

stages = [stage(filters ** 2, i) for i, filters in enumerate(range(4, 9))]
contractions = keras.Sequential(intersperse(stages, layers.MaxPooling2D(pool_size=(2, 2))))
reversed = rev(stages)
final = next(stages)
