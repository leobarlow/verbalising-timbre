#!/usr/bin/env python

"""Reduce the length and normalise the loudness of audio files.
The inputs are audio files and a metadata .json for them.
The outputs are cropped and normalised versions of the audio files.
The program uses root mean square normalisation, only considering
amplitudes above a given threshold in each audio file."""

from scipy.io import wavfile as wav
import soundfile as sf
import numpy as np
import argparse
import random
import json
import os


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-metadata', dest='metadata', action='store', type=str, required=True, help="Path for audio metadata")
parser.add_argument('-output', dest='output', action='store', type=str, required=True, help="Path for reduced audio files")
parser.add_argument('-audio', dest='audio', action='store', type=str, required=True, help="Path for audio files")
parser.add_argument('-duration', dest='duration', action='store', type=float, required=True, help="Duration for reduced files")
parser.add_argument('--loudness', dest='loudness', action='store', type=int, default=0, required=False, help="Loudness to normalise to")

def reduce_file(src, duration):
    """Reduce the length of an audio file."""
    # Load the audio file
    sr, sound = wav.read(src)
    # Take a slice of the audio array up to the required duration
    reduced = sound[:int(sr*duration)]
    return (reduced, sr)

def normalise_sound(sound, loudness):
    """Normalise the amplitude of an audio file."""
    # Root mean square normalisation
    loudness = 4000
    threshold = 1
    peak = np.max(np.abs(sound[0]))
    region = np.where(sound[0] > (peak/100)*threshold)[0]
    factor = np.sqrt((len(region) * loudness**2)/np.sum(region**2))
    normalised = (sound[0] * factor).astype(dtype=np.int16)
    return (normalised, sound[1])

def save_sound(dst, sound):
    """Save a sound to a file."""
    # Save without resampling
    sf.write(dst, sound[0], sound[1])
    return None

if __name__ == "__main__":
    args = parser.parse_args()
    duration = args.duration
    loudness = args.loudness

    # Parse the metadata
    with open(args.metadata) as metadata:
        parsed_metadata = json.load(metadata)
        keylist = list(parsed_metadata.keys())

    # Reduce the duration of each file referenced by the metadata
    print("Reducing files mentioned by %s to %s..." % (args.metadata, args.output))
    for key in keylist:
        src = os.path.join(args.audio, key + '.wav')
        dst = os.path.join(args.output, key + '.wav')
        reduced = reduce_file(src, duration)
        normalised = normalise_sound(reduced, loudness)
        save_sound(dst, normalised)
