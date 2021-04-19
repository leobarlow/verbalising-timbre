#!/usr/bin/env python

"""Extract log Mel spectral features from audio files.
The inputs are audio files and a metadata .json for them.
The outputs are features, which are written to the metadata file.
The program extracts the features using a Hamming window twice the length
of the time step. It also plots original audio files and feature vector
representations of them."""

from python_speech_features import fbank
from scipy.io import wavfile as wav
from matplotlib import pyplot as plt
import tikzplotlib as tpl
import numpy as np
import argparse
import shutil
import json
import os


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-audio', dest='audio', action='store', type=str, required=True, help="Path for audio")
parser.add_argument('-metadata', dest='metadata', action='store', type=str, required=True, help="Path for audio metadata")
parser.add_argument('-number', dest='num', action='store', type=int, required=True, help="Number of frequency bins")
parser.add_argument('-length', dest='len', action='store', type=int, required=True, help="Length of extraction window (ms)")

def extract_features(src, args):
    """Extract features from a file."""
    # Read the audio file
    sr, audio = wav.read(src)
    # Convert the length to seconds
    len = args.len/1000
    # Extract log mel-filterbank features with no preemphasis filter
    feat, energy = fbank(signal=audio, samplerate=sr, winlen=2*len, winstep=len, nfilt=args.num, nfft=4096, preemph=0, winfunc=np.hamming)
    features = np.log(feat)
    # Return features scaled between 0 and 255
    return np.clip(features.T/np.max(features) * 255, 0, 255)

def plot_spectrogram(path):
    """Generate a spectrogram for a sound file."""
    # Load the sound file
    sr, sound = wav.read(path)
    # Generate a spectrogram plot with an intensity bar
    fig, ax = plt.subplots()
    pxx, freq, t, cax = ax.specgram(sound, Fs=sr, cmap="gray")
    cbar = fig.colorbar(cax)
    cbar.set_label('Intensity (dB)')
    ax.axis("tight")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    return ax

def save_fig(name):
    """Save a figure as an image and tex."""
    plt.savefig(f'{name}.png', dpi=100, bbox_inches='tight', pad_inches=0.2)
    texdir = f'{name}_tex'
    shutil.rmtree(texdir, ignore_errors=True)
    os.makedirs(texdir)
    tpl.save(f'{texdir}/{name}.tex')
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    format = 'wav'

    # Parse the metadata
    with open(args.metadata) as metadata:
        parsed_metadata = json.load(metadata)
        keylist = list(parsed_metadata.keys())

    # Extract features from each file referenced by the metadata
    print("Extracting features from files referenced in %s..." % args.metadata)
    for key in keylist:
        src = os.path.join(args.audio, key + '.' + format)
        # Convert the features to a list
        features = extract_features(src, args).tolist()
        # Save the features in a json format
        parsed_metadata[key]['features'] = features

    # Save updates to the metadata
    print("Saving updates to %s..." % args.metadata)
    with open(args.metadata, 'w') as output:
        json.dump(parsed_metadata, output, indent = 2)

    src = 'vibrato.wav'
    # Plot a spectrogram for a file
    plot_spectrogram(src)
    save_fig('original')

    # Plot feature vectors for that file
    features = extract_features(src, args)
    plt.rcParams['image.cmap'] = 'gray'
    plt.imshow(features, origin='lower')
    plt.xlabel('Time (s)')
    plt.colorbar(label='Log intensity')
    save_fig('features')
