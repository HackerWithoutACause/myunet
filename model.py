import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

def stage(filters, index, input):
    c1 = convolution(filters, index, 'first')(input)
    d = layers.Dropout(0.1)(c1)
    return convolution(filters, index, 'second')(d)

def expansive(stright, cross, filters, index):
    transposed = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(stright)
    cat = layers.concatenate([transposed, cross])
    cv1 = convolution(filters, index, 'expansive-first')(cat)
    drop = layers.Dropout(0.2, name='Dropout-%d' % index)(cv1)
    return convolution(filters, index, 'expansive-second')(drop)

stages = [stage(16, 0, normalized)]
for i, filters in enumerate(range(5, 9)):
    pooling = layers.MaxPooling2D(pool_size=(2, 2))(stages[-1])
    stages.append(stage(2 ** filters, i + 1, pooling))

reversed = iter(reversed(stages))
last = next(reversed)

for i, con in enumerate(reversed):
    last = expansive(last, con, 2 ** (-i + 7), i)

outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(last)

model = keras.Model(inputs=[inputs], outputs=[outputs], name='unet')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
