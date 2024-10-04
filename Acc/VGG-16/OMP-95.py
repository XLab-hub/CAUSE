
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

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
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

def count_params(model):
    return tf.keras.Model.count_params(model)

def count_nonzero_params(model):
    return sum(np.count_nonzero(p) for p in model.trainable_variables)

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
datagen.fit(x_train)

from keras.optimizers import SGD
#optimization details

sgd = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = normalize(x_train, x_test)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model.fit(datagen.flow(x_train, y_train,
                                 batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=maxepoches,
                    validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=1)

_, baseline_model_accuracy = model.evaluate(x_test, y_test, verbose=0)
print('baseline acc', baseline_model_accuracy)
_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('save baseline to ', keras_file)

# Stage 2:
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
batch_size = 128
epochs = 1
validation_split = 0.1
num_images = x_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs


pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.9, final_sparsity=0.99, begin_step=0, end_step=2000)
}

start_time1 = time.time()
model_for_pruning = prune_low_magnitude(model, **pruning_params)
pruning_time = time.time() - start_time1

model_for_pruning.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
callbacks = [
tfmot.sparsity.keras.UpdatePruningStep(),
]

start_time2 = time.time()
model_for_pruning.fit(datagen.flow(x_train, y_train,
                                 batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),callbacks=callbacks,verbose=2)
Retrain_time = time.time()-start_time2


_, model_for_pruning_accuracy = model_for_pruning.evaluate(x_test, y_test, verbose=0)
print('baseline acc', baseline_model_accuracy)
print('pruning acc:', model_for_pruning_accuracy)

# Stage 3:
def get_gzipped_model_size(file):
   _, zipped_file = tempfile.mkstemp('.zip')
   with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
       f.write(file)
   return os.path.getsize(zipped_file)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print('save to ', pruned_keras_file)
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_tflite_model = converter.convert()
_, pruned_tflite_file = tempfile.mkstemp('.tflite')
with open(pruned_tflite_file, 'wb') as f:
   f.write(pruned_tflite_model)

print('baseline acc:', baseline_model_accuracy)
print('pruning acc:', model_for_pruning_accuracy)
print('baselines param', count_params(model))
print('pruning param:',count_nonzero_params(model_for_pruning))
print('Save TFLite to:', pruned_tflite_file)
print("gzip baseline Keras size: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("gzip pruningKeras size: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
print("gzip pruning TFlite size: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))
print("Retrain Time:", Retrain_time)
print("Pruning Time:", pruning_time)

print("----------------------------------------------------------------------------------------")

# train_epochs = 30
batch_size = 512
validation_split = 0.1
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

model.fit(datagen.flow(x_train, y_train,
                                 batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=1,
                    validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)

"""CAUSE-UCDP Starts """
num_train_samples = len(x_train)
num_test_samples = len(x_test)

indices = np.arange(num_train_samples)
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

num_groups = 100
min_group_size = 10
max_group_size = 10000

total_group_size = np.random.randint(min_group_size, max_group_size + 1, num_groups)
total_group_size = (total_group_size / np.sum(total_group_size)) * num_train_samples
total_group_size = np.round(total_group_size).astype(int)
total_group_size[total_group_size < min_group_size] = min_group_size
total_group_size[total_group_size > max_group_size] = max_group_size
start_index = 0
train_groups_images = []
train_groups_labels = []
for group_size in total_group_size:
    end_index = start_index + group_size
    train_groups_images.append(x_train[start_index:end_index])
    train_groups_labels.append(y_train[start_index:end_index])
    start_index = end_index

for i, group_images in enumerate(train_groups_images):
    print(f"User {i + 1} size:", len(group_images))

def cluster_data(train_groups_images, s):
    n = len(train_groups_images)
    Z = sum(len(group_images) for group_images in train_groups_images)
    shard_capacity = Z / s

    shards = [[] for _ in range(s)]
    shard_sizes = [0] * s
    for group_index, group_images in enumerate(train_groups_images):
        min_shard_index = np.argmin(shard_sizes)
        if shard_sizes[min_shard_index] + len(group_images) <= shard_capacity:
            shards[min_shard_index].append(group_index)
            shard_sizes[min_shard_index] += len(group_images)

    for i, shard in enumerate(shards):
        print(f"The {i+1}th shard including", shard)

# cluster_data(train_groups_images, s)
def cluster_data(train_groups_images, s):
    n = len(train_groups_images)
    Z = sum(len(group_images) for group_images in train_groups_images)
    sorted_groups = sorted(range(n), key=lambda x: len(train_groups_images[x]), reverse=True)
    shards = [[] for _ in range(s)]
    shard_sizes = [0] * s

    for group_idx in sorted_groups:
        best_shard = min(range(s), key=lambda x: shard_sizes[x])
        shards[best_shard].append(group_idx)
        shard_sizes[best_shard] += len(train_groups_images[group_idx])
    return shards

s = 16  # Assume 4 shards
shards = cluster_data(train_groups_images, s)
for i, shard in enumerate(shards):
    print(f"The{i + 1}th shard including：", shard)
selected_group_indices = shards[0]
selected_train_images = np.concatenate([train_groups_images[idx] for idx in selected_group_indices])
selected_train_labels = np.concatenate([train_groups_labels[idx] for idx in selected_group_indices])
# model.fit(selected_train_images, selected_train_labels, epochs=10, batch_size=256, validation_split=0.1)

"""CAUSE-UCDP Ends"""
batch_size = 512
accuracy = []
epochs_of_interest = np.arange(0,65,10)
model.fit(datagen.flow(selected_train_images, selected_train_labels,
                                 batch_size=batch_size),
                    steps_per_epoch=selected_train_images.shape[0] // batch_size,
                    epochs=1,
                    validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)

for epoch in range(1, 61):
    model.fit(datagen.flow(selected_train_images, selected_train_labels,
                                 batch_size=batch_size),
                    steps_per_epoch=selected_train_images.shape[0] // batch_size,
                    epochs=1,
                    validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)
    if epoch in epochs_of_interest:
        _, model_acc = model.evaluate(x_test, y_test, verbose=0)
        accuracy.append(model_acc)

print('target accuracy:',accuracy)