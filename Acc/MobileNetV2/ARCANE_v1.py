import numpy as np
from tensorflow.keras.datasets import cifar10
import tempfile
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
import zipfile
import time 

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
class_data = [[] for _ in range(10)]
class_labels = [[] for _ in range(10)]
for image, label in zip(train_images, train_labels):
    class_data[label[0]].append(image)
    class_labels[label[0]].append(label)
for i in range(10):
    class_data[i] = np.array(class_data[i])
    class_labels[i] = np.array(class_labels[i])

selected_classes = [0,1,2,3,4,5,6,7,8,9]
selected_train_data = np.concatenate([class_data[i] for i in selected_classes], axis=0)
selected_train_labels = np.concatenate([class_labels[i] for i in selected_classes], axis=0)

def MobileNetV2(input_shape=(32, 32, 3), num_classes=10):
    input_tensor = keras.Input(shape=input_shape)
    
    # Entry flow
    x = _conv_block(input_tensor, filters=32, kernel_size=(3, 3), strides=(1, 1))
    x = _inverted_residual_block(x, filters=16, strides=1, expansion=1)
    x = _inverted_residual_block(x, filters=24, strides=2, expansion=6)
    x = _inverted_residual_block(x, filters=24, strides=1, expansion=6)
    x = _inverted_residual_block(x, filters=32, strides=2, expansion=6)
    x = _inverted_residual_block(x, filters=32, strides=1, expansion=6)
    x = _inverted_residual_block(x, filters=32, strides=1, expansion=6)
    x = _inverted_residual_block(x, filters=64, strides=2, expansion=6)
    x = _inverted_residual_block(x, filters=64, strides=1, expansion=6)
    x = _inverted_residual_block(x, filters=64, strides=1, expansion=6)
    x = _inverted_residual_block(x, filters=64, strides=1, expansion=6)

    # Middle flow
    for _ in range(4):
        x = _inverted_residual_block(x, filters=96, strides=1, expansion=6)

    # Exit flow
    x = _inverted_residual_block(x, filters=160, strides=2, expansion=6)
    x = _inverted_residual_block(x, filters=160, strides=1, expansion=6)
    x = _inverted_residual_block(x, filters=320, strides=1, expansion=6)
    
    # Final layers
    x = keras.layers.Conv2D(1280, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(input_tensor, x, name='mobilenetv2')

    return model

def _conv_block(inputs, filters, kernel_size, strides):
    x = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    return keras.layers.ReLU()(x)

def _inverted_residual_block(inputs, filters, strides, expansion):
    # Expansion phase
    expand_channels = expansion * inputs.shape[-1]
    x = keras.layers.Conv2D(expand_channels, kernel_size=1, padding='same', use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    # Depthwise convolution
    x = keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    # Projection phase
    x = keras.layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)

    # Skip connection if input and output shapes are the same and strides is 1
    if inputs.shape[-1] == filters and strides == 1:
        x = keras.layers.Add()([inputs, x])
    return x

model = MobileNetV2(input_shape=(32, 32, 3), num_classes=100)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

accu = []
eoi = np.arange(0,51,50)
for e in range(1,51):
    model.fit(selected_train_data, selected_train_labels, batch_size= 256,epochs=1, validation_split=0.1, verbose=2)
    if e in eoi:
        _, ac = model.evaluate(test_images, test_labels, verbose=0)
        accu.append(ac)
print('initial acc', accu)
