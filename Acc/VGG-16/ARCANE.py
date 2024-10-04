
import numpy as np
import tensorflow_model_optimization as tfmot
import tempfile
import os
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import datasets, layers, models
import os
import zipfile
import time

# import tf_keras as keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from tensorflow.keras.layers import Lambda
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
import tensorflow

train=True
num_classes = 100
weight_decay = 0.0005
x_shape = [32,32,3]

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential

def vgg_16(input_shape=(32, 32, 3), num_classes=100, weight_decay=0.0005, name='vgg_16'):
    inputs = keras.layers.Input(shape=input_shape)
    
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='conv1_1')(inputs)
    x = keras.layers.BatchNormalization(name='bn1_1')(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='conv1_2')(x)
    x = keras.layers.BatchNormalization(name='bn1_2')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='conv2_1')(x)
    x = keras.layers.BatchNormalization(name='bn2_1')(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='conv2_2')(x)
    x = keras.layers.BatchNormalization(name='bn2_2')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='conv3_1')(x)
    x = keras.layers.BatchNormalization(name='bn3_1')(x)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='conv3_2')(x)
    x = keras.layers.BatchNormalization(name='bn3_2')(x)
    x = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='conv3_3')(x)
    x = keras.layers.BatchNormalization(name='bn3_3')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
    
    x = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='conv4_1')(x)
    x = keras.layers.BatchNormalization(name='bn4_1')(x)
    x = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='conv4_2')(x)
    x = keras.layers.BatchNormalization(name='bn4_2')(x)
    x = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='conv4_3')(x)
    x = keras.layers.BatchNormalization(name='bn4_3')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool4')(x)
    
    x = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='conv5_1')(x)
    x = keras.layers.BatchNormalization(name='bn5_1')(x)
    x = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='conv5_2')(x)
    x = keras.layers.BatchNormalization(name='bn5_2')(x)
    x = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='conv5_3')(x)
    x = keras.layers.BatchNormalization(name='bn5_3')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool5')(x)
    
    x = keras.layers.Dropout(0.5, name='dropout')(x)
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='fc1')(x)
    x = keras.layers.BatchNormalization(name='bn_fc1')(x)
    x = keras.layers.Dropout(0.5, name='dropout_fc1')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    return model

model = vgg_16(input_shape=(32,32,3), num_classes=100, weight_decay=0.0005)

def normalize(X_train,X_test):
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    print(mean)
    print(std)
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test

def normalize_production(x):
    mean = 121.936
    std = 68.389
    return (x-mean)/(std+1e-7)

def predict(x,normalize=True,batch_size=50):
    if normalize:
        x = normalize_production(x)
    return model.predict(x,batch_size)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

class_data = [[] for _ in range(100)]
class_labels = [[] for _ in range(100)]

for image, label in zip(x_train, y_train):
    class_data[label[0]].append(image)
    class_labels[label[0]].append(label)

for i in range(100):
    class_data[i] = np.array(class_data[i])
    class_labels[i] = np.array(class_labels[i])

selected_classes = list(range(100)) 

selected_train_data = np.concatenate([class_data[i] for i in selected_classes], axis=0)
selected_train_labels = np.concatenate([class_labels[i] for i in selected_classes], axis=0)

selected_train_data = selected_train_data.astype('float32')
selected_train_labels = selected_train_labels.astype('float32')
selected_train_data, selected_train_labels = normalize(selected_train_data, selected_train_labels)
selected_train_labels = keras.utils.to_categorical(selected_train_labels, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = normalize(x_train, x_test)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

batch_size = 128
maxepoches = 1
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20

def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))
reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

#data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(selected_train_data)

from keras.optimizers import SGD
#optimization details

sgd = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

model.fit(x_train, y_train,
                    batch_size=batch_size,
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=maxepoches,
                    validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=1)

print("----------------------------------------------------------------------------------------")

batch_size = 512
validation_split = 0.1

model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

accu = []
eoi = np.arange(0,61,10)
for e in range(1,61):
    model.fit(selected_train_data, selected_train_labels,
                    batch_size=batch_size,
                    steps_per_epoch=selected_train_data.shape[0] // batch_size,
                    epochs=1,
                    validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)
    if e in eoi:
        _, baseline_model_accuracy = model.evaluate(x_test, y_test, verbose=0)
        accu.append(baseline_model_accuracy)
print('Stage 1 accuracy',accu)