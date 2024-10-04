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

selected_classes = [0,1,2]

selected_train_data = np.concatenate([class_data[i] for i in selected_classes], axis=0)
selected_train_labels = np.concatenate([class_labels[i] for i in selected_classes], axis=0)

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras.layers.Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2c')(x)

    x = keras.layers.add([x, input_tensor])
    x = keras.layers.Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras.layers.Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = keras.layers.Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = keras.layers.BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x

def resnet34(input_shape=(28, 28, 1), num_classes=10):
    input_tensor = keras.layers.Input(shape=input_shape)

    x = keras.layers.ZeroPadding2D((3, 3))(input_tensor)
    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = keras.layers.BatchNormalization(name='bn_conv1')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = keras.layers.Dense(num_classes, activation='softmax', name='fc')(x)

    model = keras.models.Model(input_tensor, x, name='resnet34')
    return model

model = resnet34(input_shape=(32, 32, 3), num_classes=10)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


accu = []
eoi = np.arange(0,81,10)
for e in range(1,81):
    model.fit(selected_train_data, selected_train_labels, batch_size= 256,epochs=1, validation_split=0.1, verbose=2)
    if e in eoi:
        _, ac = model.evaluate(test_images, test_labels, verbose=0)
        accu.append(ac)
print('initial acc', accu)