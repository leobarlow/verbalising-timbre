#!/usr/bin/env python

"""Test a trained neural network's ability to predict metaphorical timbre judgements.
The inputs are training, validation and test .json files (which contain metadata for
each sound in the dataset, with extracted features and metaphorical judgements), and
a trained neural network. 
The outputs are error statistics and plots for unweighted and weighted network 
accuracies on both the validation and test sets."""

from matplotlib import pyplot as plt
from tensorflow import keras
import tikzplotlib as tpl
import tensorflow as tf
import numpy as np
import argparse
import json
import sys
import os


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-train', dest='train', action='store', type=str, required=True, help="Path for training metadata")
parser.add_argument('-valid', dest='valid', action='store', type=str, required=True, help="Path for validation metadata")
parser.add_argument('-test', dest='test', action='store', type=str, required=True, help="Path for testing metadata")

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

def metric_names(metadata_fname):
    """Return an alphabetically sorted list of the evaluated scales."""
    with open(metadata_fname, 'r') as f:
        metadata = json.load(f)
    sound_name_list = sorted(metadata.keys())
    eval_name_list = sorted(metadata[sound_name_list[0]]['evaluations'].keys())
    return eval_name_list

def accuracy_results(timbre_model, dataset, x, y_true, metric_name_list):
    """Print accuracy results for a model."""
    y_pred = timbre_model.predict(x)
    y_true_val = np.argmax(y_true, axis=-1)
    y_pred_val = np.argmax(y_pred, axis=-1)
    print()
    print(f'Dataset: {dataset}')
    print(f'Dataset size = {x.shape[0]}')
    for i, metric in enumerate(metric_name_list):
        avg = np.mean(np.abs(y_pred_val[:,i] - y_true_val[:,i]))
        print(f'Average error for {metric} = {avg:.2f}')
        rms = np.sqrt(np.mean(np.abs(y_pred_val[:,i] - y_true_val[:,i]) ** 2))
        print(f'RMS error for {metric} = {rms:.2f}')
        for example in range(5):
            print('true, pred =', y_true_val[example,i], y_pred_val[example,i])

def accuracy_histogram(timbre_model, dataset, x, y_true, metric_name_list):
    """Plot prediction errors for a model."""
    y_pred = timbre_model.predict(x)
    y_true_val = np.argmax(y_true, axis=-1)
    y_pred_val = np.argmax(y_pred, axis=-1)
    fig, axs = plt.subplots(len(metric_name_list), 1, constrained_layout=True)
    for i, metric in enumerate(metric_name_list):
        hist = np.histogram(y_pred_val[:,i] - y_true_val[:,i], bins=19, range=(-9, 10))
        axs[i].bar(x=range(-9, 10), height=hist[0], edgecolor='black', color='grey', linewidth=0.1)
        axs[i].set_title(metric)
        axs[i].set_ylabel('Freq')
        axs[i].set_xticks(np.arange(-9, 10))
        axs[i].tick_params('x', which='both', length=0)
    axs[-1].set_xlabel('Prediction error')
    return plt

def weighted_accuracy_histogram(timbre_model, dataset, x, y_true, metric_name_list):
    """Plot weighted prediction errors for a model."""
    y_pred = timbre_model.predict(x)
    y_true_val = np.argmax(y_true, axis=-1)
    y_pred_val = np.argmax(y_pred, axis=-1)
    fig, axs = plt.subplots(len(metric_name_list), 1, constrained_layout=True)
    for i, metric in enumerate(metric_name_list):
        # Calculate weights based on evaluation score rarity
        score_freq = np.bincount(y_true_val[:,i])
        print(metric, score_freq)
        score_frac = score_freq / np.sum(score_freq)
        weighting = 1 / (score_frac + 1E-6)
        weights = weighting[y_true_val[:,i]]
        # Calculate the histogram bar sizes
        hist = np.histogram(y_pred_val[:,i] - y_true_val[:,i], weights=weights, density=True, bins=19, range=(-9, 10))
        print(f'Probability of exact {metric} result = {np.sum(hist[0][9]):.2f}')
        print(f'Probability of within +/-1 {metric} result = {np.sum(hist[0][8:11]):.2f}')
        print(f'Probability of within +/-2 {metric} result = {np.sum(hist[0][7:12]):.2f}')
         # Plot the histogram bar graph
        axs[i].bar(x=range(-9, 10), height=hist[0], edgecolor='black', color='grey', linewidth=0.1)
        axs[i].set_title(metric)
        axs[i].set_ylabel('Probability')
        axs[i].set_xticks(np.arange(-9, 10))
        axs[i].tick_params('x', which='both', length=0)
    axs[-1].set_xlabel('Prediction error')
    return plt

def plot_prediction(timbre_model, features, labels, index):
    """Plot scale bar charts for a prediction."""
    predictions = timbre_model.predict(features)
    fig, axs = plt.subplots(1, 3, sharey=True, constrained_layout=True)
    for i, metric in enumerate(metric_name_list):
        axs[i].bar(x=range(10), height=predictions[index][i], edgecolor='black', color='grey', linewidth=0.1)
        axs[i].set_xlabel(metric)
        axs[i].set_xticks(range(10))
        axs[i].tick_params('x', which='both', length=0)
    axs[0].set_ylabel('Probability')
    return plt

def save_fig(name):
    """Save a figure as an image and tex."""
    plt.savefig(f'{name}.png', dpi=100, bbox_inches='tight', pad_inches=0.2)
    tpl.save(f'{name}.tex')
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()

    # Load the model
    timbre_model = keras.models.load_model('trained_model.h5')
    metric_name_list = metric_names(args.valid)

    # Parse the validation set
    x_valid, y_valid = make_xy(args.valid)

    # Plot a single prediction
    plot_prediction(timbre_model, x_valid, y_valid, index=13)
    save_fig('prediction')

    # Plot the validation set accuracies
    accuracy_results(timbre_model, 'Validation', x_valid, y_valid, metric_name_list)
    accuracy_histogram(timbre_model, 'Validation', x_valid, y_valid, metric_name_list)
    save_fig('valid_accuracy')
    weighted_accuracy_histogram(timbre_model, 'Validation', x_valid, y_valid, metric_name_list)
    save_fig('valid_weighted_accuracy')

    # Plot the test set accuracies
    x_test, y_test = make_xy(args.test)
    accuracy_results(timbre_model, 'Test', x_test, y_test, metric_name_list)
    accuracy_histogram(timbre_model, 'Test', x_test, y_test, metric_name_list)
    save_fig('test_accuracy')
    weighted_accuracy_histogram(timbre_model, 'Test', x_test, y_test, metric_name_list)
    save_fig('test_weighted_accuracy')
