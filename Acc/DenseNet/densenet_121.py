import tempfile
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
import zipfile
import time 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cifar100 = keras.datasets.cifar100
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
#train_images, test_images = train_images / 255.0, test_images / 255.0
train_images, test_images = train_images[:1000] / 255.0, test_images[:200] / 255.0
train_labels, test_labels = train_labels[:1000], test_labels[:200]
# DenseNet-121
def dense_block(input_tensor, num_layers, growth_rate, name):
    x = input_tensor
    for i in range(num_layers):
        x = conv_block(x, growth_rate, name=name + '_layer{}'.format(i + 1))
    return x

def conv_block(input_tensor, growth_rate, name):
    x1 = keras.layers.BatchNormalization(axis=-1, name=name + '_bn1')(input_tensor)
    x1 = keras.layers.Activation('relu', name=name + '_relu1')(x1)
    x1 = keras.layers.Conv2D(4 * growth_rate, (1, 1), padding='same', name=name + '_conv1')(x1)
    x1 = keras.layers.BatchNormalization(axis=-1, name=name + '_bn2')(x1)
    x1 = keras.layers.Activation('relu', name=name + '_relu2')(x1)
    x1 = keras.layers.Conv2D(growth_rate, (3, 3), padding='same', name=name + '_conv2')(x1)
    x = keras.layers.concatenate([input_tensor, x1], axis=-1, name=name + '_concat')
    return x

def transition_layer(input_tensor, compression_factor, name):
    num_filters = int(input_tensor.shape[-1] * compression_factor)
    x = keras.layers.BatchNormalization(axis=-1, name=name + '_bn')(input_tensor)
    x = keras.layers.Activation('relu', name=name + '_relu')(x)
    x = keras.layers.Conv2D(num_filters, (1, 1), padding='same', name=name + '_conv')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name=name + '_pool')(x)
    return x

def DenseNet121(input_shape=(32, 32, 3), classes=100, name='densenet121'):
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inputs)
    x = keras.layers.BatchNormalization(axis=-1, name='bn_conv1')(x)
    x = keras.layers.Activation('relu', name='relu1')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(x)

    # Dense Block 1
    x = dense_block(x, num_layers=6, growth_rate=32, name='dense_block1')
    x = transition_layer(x, compression_factor=0.5, name='transition_layer1')

    # Dense Block 2
    x = dense_block(x, num_layers=12, growth_rate=32, name='dense_block2')
    x = transition_layer(x, compression_factor=0.5, name='transition_layer2')

    # Dense Block 3
    x = dense_block(x, num_layers=24, growth_rate=32, name='dense_block3')
    x = transition_layer(x, compression_factor=0.5, name='transition_layer3')

    # Dense Block 4
    x = dense_block(x, num_layers=16, growth_rate=32, name='dense_block4')

    x = keras.layers.GlobalAveragePooling2D(name='pool5')(x)
    x = keras.layers.Dense(classes, activation='softmax', name='fc')(x)

    model = keras.models.Model(inputs, x, name=name)
    return model
model = DenseNet121(input_shape=(32, 32, 3), classes=100, name='densenet121')
model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=1, validation_split=0.1)
#_, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)
#print('test-baseline：', baseline_model_accuracy)
_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=tempfile.mkdtemp()),
]

# model = keras.models.load_model(keras_file)
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.60,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=2000)
}

def get_gzipped_model_size(file):
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file)

#print("gzip pruned baselines Keras size: %.2f bytes" % (get_gzipped_model_size(keras_file)))
#print("gzip pruned Keras size: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file_stripped)))
#print("gzip pruned TFlite size: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))
print("----------------------------------------------------------------------------------------")

train_epochs = 1
batch_size = 64
validation_split = 0.1

model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=batch_size, epochs=train_epochs, validation_split=validation_split)
_, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)

target_sparsity_values = np.arange(0.1, 1.0, 0.2)
initial_sparsity = np.arange(0.0, 0.9, 0.2, dtype=np.float32)
final_sparsity = np.arange(0.2, 1.1, 0.2, dtype=np.float32)
final_sparsity[-1] = 0.99

model_for_pruning_accuracies = []
pruned_keras_files = []
pruned_keras_file_sizes = []
train_time = []
pruning_times = []
pruned_params = []
degradation_acc = []
degradation_file = []
degradation_params = []
losses = []

baseline_model_accuracy 

def count_params(model):
    return tf.keras.Model.count_params(model)

def count_nonzero_params(model):
    return sum(np.count_nonzero(p) for p in model.trainable_variables)

retrain_epochs = 1
retrain_batch_size = 128

num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / 100).astype(np.int32) * retrain_epochs

for i in range(len(initial_sparsity)):
    model_for_pruning = tf.keras.models.clone_model(model)
    model_for_pruning.set_weights(model.get_weights())
    params = count_params(model)
    origional_size = get_gzipped_model_size(keras_file)
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=initial_sparsity[i], final_sparsity=final_sparsity[i], begin_step=0, end_step=end_step)
    }

    # Record Pruning Time
    start_time1 = time.time()
    model_for_pruning = prune_low_magnitude(model_for_pruning, **pruning_params)

    # Pruning Time
    pruning_time = time.time() - start_time1
    pruning_times.append(pruning_time)

    # Train and record training time:
    start_time2 = time.time()
    model_for_pruning.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    history = model_for_pruning.fit(train_images, train_labels, batch_size=batch_size, epochs=retrain_epochs, validation_split=validation_split, callbacks=callbacks)
    training_time = time.time()-start_time2
    train_time.append(training_time)
    _, model_for_pruning_accuracy = model_for_pruning.evaluate(test_images, test_labels, verbose=0)

    # Accuracy after pruning
    model_for_pruning_accuracies.append(model_for_pruning_accuracy)

    # Accuracy degradation after pruning
    degradation_acc.append((baseline_model_accuracy - model_for_pruning_accuracy)/baseline_model_accuracy)

    _, pruned_keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model_for_pruning, pruned_keras_file, include_optimizer=False)
    pruned_keras_files.append(pruned_keras_file)

    # Params after pruning
    pruned_params.append(count_nonzero_params(model_for_pruning))

    # Params degradation after pruning
    degradation_params.append((params - count_nonzero_params(model_for_pruning))/params)

    # File Size after pruning
    pruned_keras_file_sizes.append(get_gzipped_model_size(pruned_keras_file)/(1024*1024))

    # File size degradation after pruning
    degradation_file.append((origional_size - get_gzipped_model_size(pruned_keras_file))/origional_size)

    losses.append(history.history['loss'][-1])

plt.figure(figsize=(18, 6))
plt.subplot(1, 4, 1)
plt.plot(target_sparsity_values, model_for_pruning_accuracies, marker='o')
plt.xlabel('Target Sparsity')
plt.ylabel('Model Accuracy')
plt.title('Model Accuracy vs. Target Sparsity')
plt.subplot(1, 4, 2)
plt.plot(target_sparsity_values, pruned_keras_file_sizes, marker='o')
plt.xlabel('Target Sparsity')
plt.ylabel('Pruned Keras File Size (bytes)')
plt.title('Pruned Keras File Size vs. Target Sparsity')
plt.subplot(1, 4, 3)
plt.plot(target_sparsity_values, losses, marker='o', color='green')
plt.xlabel('Target Sparsity')
plt.ylabel('Loss')
plt.title('Loss vs. Target Sparsity')
plt.subplot(1, 4, 4)
plt.plot(target_sparsity_values, pruning_times, marker='o', color='orange')
plt.xlabel('Target Sparsity')
plt.ylabel('Pruning Time (seconds)')
plt.title('Pruning Time vs. Target Sparsity')
plt.tight_layout()
plt.show()

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