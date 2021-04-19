#!/usr/bin/env python

"""Augment audio files by creating variants of them.
The inputs are audio files and a metadata .json for them.
The outputs are copies of the files, a number of variants of each, and
an updated metadata .json.
The program creates variants for each file using levels of four
augmentation techniques: noise addition, time shifting, time stretching
and time squashing. It also plots some exaggerated augmentation spectrograms."""

from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import pyrubberband as rb
import tikzplotlib as tpl
import soundfile as sf
import numpy as np
import scipy.io
import argparse
import shutil
import json
import os


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-metadata', dest='metadata', action='store', type=str, required=True, help="Path for metadata")
parser.add_argument('-audio', dest='audio', action='store', type=str, required=True, help="Path for audio files")
parser.add_argument('-augaudio', dest='augaudio', action='store', type=str, required=True, help="Path for augmented audio files")
parser.add_argument('-augmetadata', dest='augmetadata', action='store', type=str, required=True, help="Path for augmented audio files")
parser.add_argument('-number', dest='number', action='store', type=int, required=True, help="Number of variants to create")
parser.add_argument('-max', dest='max', action='store', type=int, required=True, help="Maximum percentage difference to originals")
parser.add_argument('--depth', dest='depth', action='store', type=int, default=16, required=False, help="Audio bit depth")

def add_noise(src, depth, percent):
    """Add noise to an audio file."""
    sr, sound = wav.read(src)
    # Generate a noise array between -1 and 1
    noise = (np.random.random(len(sound)) * 2) - 1
    # Add sound to the audio file as a multiple of the bit depth and percentage
    for i in range(len(sound)):
        sound[i] = sound[i] + int(noise[i] * (2**depth)/2 * percent)
    return (sound, sr)

def shift_sound(src, number):
    """Shift the samples in an audio file."""
    sr, sound = wav.read(src)
    # Set the shift to a percentage of the sample rate
    shift = number * int(sr/100)
    # Take a slice of the audio array up to the shift point
    crop = sound[:-shift]
    # Add equivalent silence to the start of the new array
    sound = np.concatenate((np.zeros(shift, dtype=np.int16), crop))
    return (sound, sr)

def stretch_sound(src, percent):
    """Stretch an audio file."""
    sr, sound = wav.read(src)
    # Stretch the sound by the given percentage
    stretch = rb.pyrb.time_stretch(sound, sr, 1 - percent)
    # Crop the sound to the same length as the original
    sound = stretch[0:len(sound)]
    return (sound, sr)

def squish_sound(src, percent):
    """Squish an audio file."""
    sr, sound = wav.read(src)
    # Stretch the sound by the given percentage
    squish = rb.pyrb.time_stretch(sound, sr, 1 + percent)
    # Add silence to produce a sound with the same length as the original
    silence = np.zeros(len(sound) - len(squish))
    sound = np.concatenate((squish, silence))
    return (sound, sr)

def get_intervals(max, number):
    """Return augmentation intervals."""
    intervals = np.linspace(0, max, number + 1)
    return np.delete(intervals, 0)

def save_variant(dst, variant):
    """Save a variant to a file."""
    sf.write(dst, variant[0], variant[1])
    return None

def plot_spectrogram(path):
    """Generate a spectrogram for an audio file."""
    sr, sound = wav.read(path)
    fig, ax = plt.subplots()
    pxx, freq, t, cax = ax.specgram(sound, Fs=sr, cmap="gray")
    cbar = fig.colorbar(cax)
    cbar.set_label('Intensity (dB)')
    ax.axis("tight")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    return ax

def save_metadata(metadata_path, parsed_metadata):
    """Save the augmented metadata."""
    print("Saving augmented metadata to %s..." % metadata_path)
    with open(metadata_path, 'w') as output:
        json.dump(parsed_metadata, output, indent = 2)

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
    number = args.number
    max = args.max
    depth = args.depth

    print("Augmenting audio files referenced in %s..." % args.metadata)

    # Parse the metadata
    with open(args.metadata) as metadata:
        parsed_metadata = json.load(metadata)
        keylist = list(parsed_metadata.keys())

    # Copy every file
    for key in keylist:
        src = os.path.join(args.audio, key + '.wav')
        dst = os.path.join(args.augaudio, key + '.wav')
        shutil.copyfile(src, dst)

    # Create variants using noise
    for key in keylist:
        for interval in get_intervals(max, number):
            info = {}
            info['source'] = key
            info['variation'] = 'noise'
            info['interval'] = interval
            name = key + '_noise_' + '{:.2f}'.format(interval)
            src = os.path.join(args.audio, key + '.wav')
            dst = os.path.join(args.augaudio, name + '.wav')
            variant = add_noise(src, depth, interval/100)
            save_variant(dst, variant)
            parsed_metadata[name] = info

    # Create variants with shifting
    for key in keylist:
        for value in range(number):
            info = {}
            info['source'] = key
            info['variation'] = 'shift'
            info['interval'] = value + 1
            name = key + '_shift_' + '{:02}'.format(value + 1)
            src = os.path.join(args.audio, key + '.wav')
            dst = os.path.join(args.augaudio, name + '.wav')
            variant = shift_sound(src, value + 1)
            save_variant(dst, variant)
            parsed_metadata[name] = info

    # Create variants using stretching
    for key in keylist:
        for interval in get_intervals(max, number):
            info = {}
            info['source'] = key
            info['variation'] = 'stretch'
            info['interval'] = interval
            name = key + '_stretch_' + '{:.2f}'.format(interval)
            src = os.path.join(args.audio, key + '.wav')
            dst = os.path.join(args.augaudio, name + '.wav')
            variant = stretch_sound(src, interval/100)
            save_variant(dst, variant)
            parsed_metadata[name] = info

    # Create variants using squishing
    for key in keylist:
        for interval in get_intervals(max, number):
            info = {}
            info['source'] = key
            info['variation'] = 'squish'
            info['interval'] = interval
            name = key + '_squish_' + '{:.2f}'.format(interval)
            src = os.path.join(args.audio, key + '.wav')
            dst = os.path.join(args.augaudio, name + '.wav')
            variant = squish_sound(src, interval/100)
            save_variant(dst, variant)
            parsed_metadata[name] = info

    # Save the updated metadata
    save_metadata(args.augmetadata, parsed_metadata)


    src = 'demo.wav'
    # Create some exaggerated modifications
    noise = add_noise(src, depth, 0.02)[0]
    shift = shift_sound(src, 20)[0]
    stretch = stretch_sound(src, 0.2)[0]
    squish = squish_sound(src, 0.2)[0]

    # Plot the exaggerated modifications
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=False)
    pxx, freq, t, cax = ax1.specgram(noise, Fs=16000, cmap="gray")
    pxx, freq, t, cax = ax2.specgram(shift, Fs=16000, cmap="gray")
    pxx, freq, t, cax = ax3.specgram(squish, Fs=16000, cmap="gray")
    pxx, freq, t, cax = ax4.specgram(stretch, Fs=16000, cmap="gray")
    ax1.title.set_text('Noise')
    ax1.set_ylabel('Frequency (Hz)')
    ax2.title.set_text('Shifting')
    ax3.title.set_text('Squashing')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_xlabel('Time (s)')
    ax4.title.set_text('Stretching')
    ax4.set_xlabel('Time (s)')
    ax1.axis('tight')
    ax2.axis('tight')
    ax3.axis('tight')
    ax4.axis('tight')
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    cbar = fig.colorbar(cax, cax=cbar_ax)
    cbar.set_label('Intensity (dB)')

    save_fig('augmentation')
