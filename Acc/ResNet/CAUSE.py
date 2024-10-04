import tempfile
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
import zipfile
import time 
from keras.optimizers import Adam
# Fix data shards = 5

cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images_split = np.split(train_images, 1)
train_labels_split = np.split(train_labels, 1)

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

def resnet34(input_shape=(32, 32, 3), num_classes=10):
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


"""CAUSE-UCDP Starts """
s = 4  
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
        print(f"The {i+1}th shard including:", shard)


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

s = 2  
shards = cluster_data(train_groups_images, s)
for i, shard in enumerate(shards):
    print(f"The {i+1}th shard including:", shard)


selected_group_indices = shards[0]
selected_train_images = np.concatenate([train_groups_images[idx] for idx in selected_group_indices])
selected_train_labels = np.concatenate([train_groups_labels[idx] for idx in selected_group_indices])

"""CAUSE-UCDP Ends"""

history1 = model.fit(train_images, train_labels, batch_size= 256,epochs=1, validation_split=0.1, verbose=2)

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('begin pruning')

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
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

history2 = model.fit(selected_train_images,selected_train_labels, validation_split=0.1, epochs=10,batch_size=128,callbacks=callbacks,verbose=2)
history3 = model.evaluate(test_images, test_labels, verbose=0)
print('evaluate', history3)
print("stage 1 val acc:", history1.history['val_accuracy'])
print("stage 2 val acc:", history2.history['val_accuracy'])
merged_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
print("Merged val acc:", merged_acc )