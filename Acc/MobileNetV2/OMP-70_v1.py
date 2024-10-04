import tempfile
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
import zipfile
import time 
from tensorflow.keras.optimizers import Adam
# Fix data shards = 5

cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images_split = np.split(train_images, 1)
train_labels_split = np.split(train_labels, 1)

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


"""CAUSE-UCDP Starts """
s = 1  
num_train_samples = len(train_images)
num_test_samples = len(test_images)
indices = np.arange(num_train_samples)
np.random.shuffle(indices)
train_images = train_images[indices]
train_labels = train_labels[indices]

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
    train_groups_images.append(train_images[start_index:end_index])
    train_groups_labels.append(train_labels[start_index:end_index])
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
        print(f"The {i+1}th shard including：", shard)
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

s = 1 
shards = cluster_data(train_groups_images, s)
for i, shard in enumerate(shards):
    print(f"The {i + 1}th shard including:", shard)

selected_group_indices = shards[0]
selected_train_images = np.concatenate([train_groups_images[idx] for idx in selected_group_indices])
selected_train_labels = np.concatenate([train_groups_labels[idx] for idx in selected_group_indices])
# model.fit(selected_train_images, selected_train_labels, epochs=10, batch_size=256, validation_split=0.1)

"""CAUSE-UCDP Ends"""
accu = []
history1 = model.fit(train_images, train_labels, batch_size= 256,epochs=1, validation_split=0.1, verbose=2)

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep()
  # tfmot.sparsity.keras.PruningSummaries(log_dir=tempfile.mkdtemp()),
]
model = keras.models.load_model(keras_file)
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.60,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=2000)
}
accuracy= []
model = prune_low_magnitude(model, **pruning_params)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epochs_of_interest = np.arange(0,50,10)
# for epoch in range(1, 51):
history2 = model.fit(selected_train_images, selected_train_labels, validation_split=0.1, epochs=50, batch_size=128, callbacks=callbacks, verbose=2)
print("stage 1 val acc:", history1.history['val_accuracy'])
print("stage 2 val acc:", history2.history['val_accuracy'])
merged_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
print("merged val acc:", merged_acc )