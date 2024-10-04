import tempfile
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
import zipfile
import time 

cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
#train_images, test_images = train_images[:1000] / 255.0, test_images[:200] / 255.0
#train_labels, test_labels = train_labels[:1000], test_labels[:200]
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

# (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
# train_images = train_images / 255.0
# test_images = test_images / 255.0

model = MobileNetV2(input_shape=(32, 32, 3), num_classes=100)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1, validation_split=0.1,verbose=2)
_, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print('test-baseline acc:', baseline_model_accuracy)

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=tempfile.mkdtemp()),
]
model = keras.models.load_model(keras_file)

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.60,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=2000)
}
model_for_pruning = prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
model_for_pruning.fit(train_images, train_labels, batch_size=256, epochs=10, validation_split=0.1, callbacks=callbacks,verbose=2)

_, pruned_model_accuracy = model_for_pruning.evaluate(test_images, test_labels, verbose=0)
print('acc after pruning', pruned_model_accuracy)
_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_pruning, pruned_keras_file, include_optimizer=False)

# Stage 2:
_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_pruning, pruned_keras_file, include_optimizer=False)
print('save pruned Keras to:', pruned_keras_file)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
_, pruned_keras_file_stripped = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_export, pruned_keras_file_stripped, include_optimizer=False)
print('save pruned Keras to:', pruned_keras_file_stripped)

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_tflite_model = converter.convert()

_, pruned_tflite_file = tempfile.mkstemp('.tflite')
with open(pruned_tflite_file, 'wb') as f:
    f.write(pruned_tflite_model)
print('save pruned TFLite to', pruned_tflite_file)

def get_gzipped_model_size(file):
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file)

print("gzip baseline Keras model size: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("gzip pruned Keras size: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file_stripped)))
print("gzip pruned TFlite size: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))
print("----------------------------------------------------------------------------------------")

train_epochs = 50
batch_size = 128
validation_split = 0.1

model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=batch_size, epochs=train_epochs, validation_split=validation_split, verbose = 2)
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

retrain_epochs = 20
retrain_batch_size = 80

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
    history = model_for_pruning.fit(train_images, train_labels, batch_size=batch_size, epochs=retrain_epochs, validation_split=validation_split, callbacks=callbacks,verbose = 2)
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
