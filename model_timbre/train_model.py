#!/usr/bin/env python

"""Train a neural network to predict metaphorical timbre judgements.
The inputs are training and validation .json files, which contain
metadata for each sound in the dataset, with extracted features and
metaphorical judgements.
The output is trained_model.h5, which contains the network with the
weights that performed best on the validation set, and can be loaded
to make predictions."""

from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras
import tikzplotlib as tpl
import tensorflow as tf
import numpy as np
import argparse
import time
import json
import sys
import os


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-train', dest='train', action='store', type=str, required=True, help="Path for training metadata")
parser.add_argument('-valid', dest='valid', action='store', type=str, required=True, help="Path for validation metadata")
parser.add_argument('--epochs', dest='epochs', action='store', type=int, default=10, required=False, help="Number of training epochs")
parser.add_argument('--batch', dest='batch', action='store', type=int, default=128, required=False, help="Training batch size")
parser.add_argument('--rate', dest='rate', action='store', type=float, default=0.0005, required=False, help="Learning rate")

def model1():
    """Build a basic convnet."""
    input = layers.Input(shape=input_shape[1:], dtype=np.float)
    x = layers.Conv1D(10, 5, activation='relu', name='CONV_1')(input)
    x = layers.Conv1D(10, 5, activation='relu', name='CONV_2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='MAXPOOL_2')(x)
    x = layers.Conv1D(10, 5, activation='relu', name='CONV_3')(x)
    x = layers.Flatten()(x)
    xx = []
    for i in range(3):
        xi = layers.Dense(10, activation='softmax', name=f'DENSE_SOFTMAX_{i}')(x)
        xi = layers.Reshape((1,-1))(xi)
        xx.append(xi)
    output = layers.Concatenate(axis=1)(xx)
    return(keras.Model(input, output))

def model2():
    """Build a convnet with dropout and batch normalisation."""
    input = layers.Input(shape=input_shape[1:], dtype=np.float)
    x = layers.Conv1D(10, 5, activation='relu', name='CONV_1')(input)
    x = layers.Conv1D(10, 5, activation='relu', name='CONV_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.MaxPooling1D(pool_size=2, name='MAXPOOL_2')(x)
    x = layers.Conv1D(10, 5, activation='relu', name='CONV_3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Flatten()(x)
    xx = []
    for i in range(3):
        xi = layers.Dense(10, activation='softmax', name=f'DENSE_SOFTMAX_{i}')(x)
        xi = layers.Reshape((1,-1))(xi)
        xx.append(xi)
    output = layers.Concatenate(axis=1)(xx)
    return(keras.Model(input, output))

def model3():
    """Build a fully connected model with a convolutional input layer."""
    input = layers.Input(shape=input_shape[1:], dtype=np.float)
    x = layers.Conv1D(10, 5, activation='relu', name='CONV_1')(input)
    x = layers.MaxPooling1D(pool_size=2, name='MAXPOOL_1')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(80, activation='relu', name='DENSE_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(80, activation='relu', name='DENSE_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    xx = []
    for i in range(3):
        xi = layers.Dense(10, activation='softmax', name=f'DENSE_SOFTMAX_{i}')(x)
        xi = layers.Reshape((1,-1))(xi)
        xx.append(xi)
    output = layers.Concatenate(axis=1)(xx)
    return keras.Model(input, output)

def plot_epochs(hist):
    """Plot loss by number of training epochs."""
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper right')
    return plt

def make_xy(metadata_fname):
    """Create a Keras x and y dataset from a given metadata file."""
    # Read and parse the metadata
    with open(metadata_fname, 'r') as f:
        metadata = json.load(f)
    sound_name_list = sorted(metadata.keys())

    # Convert acoustic features into 3D input arrays
    x = np.array([metadata[name]['features'] for name in sound_name_list], dtype=np.float)

    # Convert evaluation data into one-hot arrays
    eval_name_list = sorted(metadata[sound_name_list[0]]['evaluations'].keys())
    y = []
    for name in sound_name_list:
        evals = [metadata[name]['evaluations'][e] for e in eval_name_list]
        one_hots = keras.utils.to_categorical(evals, num_classes=10)
        y.append(tf.constant(one_hots, shape=[3, 10]))
    y = np.array(y)
    return x, y

def save_fig(name):
    """Save a figure as an image and tex."""
    plt.savefig(f'{name}.png', dpi=100, bbox_inches='tight', pad_inches=0.2)
    tpl.save(f'{name}.tex')
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()

    # Define the netwtork inputs
    num_samples = 20
    num_bins = 20
    input_shape = (args.batch, num_samples, num_bins)

    # Build and compile a model
    timbre_model = model2()
    adam = keras.optimizers.Adam(learning_rate=args.rate)
    timbre_model.compile(optimizer=adam, loss="categorical_crossentropy")
    timbre_model.summary()

    # Parse the training and validation sets
    x_train, y_train = make_xy(args.train)
    x_valid, y_valid = make_xy(args.valid)

    # Train the model
    checkpoint = ModelCheckpoint(filepath='trained_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    start_time = time.time()
    hist = timbre_model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=args.batch, epochs=args.epochs,
        verbose=2, callbacks=[checkpoint])
    end_time = time.time()
    print(f'Training time = {end_time - start_time:.1f}s')

    # Plot loss by epoch
    plot_epochs(hist)
    save_fig('training_loss')
