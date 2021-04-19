#!/usr/bin/env python

"""Create training, validation and test samples from an audio metadata file.
The inputs are a collated metadata .json.
The outputs are training, validation and test .json files containing
subsets of the metadata.
The program finds the most common value of a specified attribute (e.g. pitch)
in the collated metadata, then filters out metadata entries with other values of
that attribute when creating the samples. If a quality is specified (e.g.
percussive), entries with that quality are filtered out too. It was originally
designed to include timbres with multiple dynamics in the samples, while
making sure variants of the same timbre didn't end up in different samples."""

from collections import deque
import argparse
import random
import json
import os


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-metadata', dest='metadata', action='store', type=str, required=True, help="Path for metadata")
parser.add_argument('-train', dest='train', action='store', type=str, required=True, help="Path for training sample")
parser.add_argument('-valid', dest='valid', action='store', type=str, required=True, help="Path for validation sample")
parser.add_argument('-test', dest='test', action='store', type=str, required=True, help="Path for testing sample")
parser.add_argument('-total', dest='total', action='store', type=str, required=True, help="Path for collated samples")
parser.add_argument('-size', dest='size', action='store', type=int, required=True, help="Total size of samples")
parser.add_argument('-ratio', dest='ratio', action='store', type=int, nargs ='+', required=True, help="Training-validation-testing ratio")
parser.add_argument('-cap', dest='cap', action='store', type=int, required=False, help="Cap for velocity variants")
parser.add_argument('--attribute', dest='attribute', action='store', type=str, default='pitch', required=False, help="Attribute to hold consistent")
parser.add_argument('--quality', dest='quality', action='store', type=str, required=False, help="Quality to filter out")

def create_samples(args, mode):
    """Split the metadata into distinct samples."""
    with open(args.metadata) as metadata:
        # Parse the metadata
        parsed_metadata = json.load(metadata)
        keylist = list(parsed_metadata.keys())

        filteredkeys = []
        for key in keylist:
            # Filter out sounds that don't have the required attribute value
            if parsed_metadata[key][args.attribute] == mode:
                # If a quality is specified, filter it out too
                if args.quality:
                    if not any(args.quality in s for s in parsed_metadata[key]['qualities_str']):
                        filteredkeys.append(key)
                else:
                    filteredkeys.append(key)

        # Randomise the order of the data
        random.shuffle(filteredkeys)

        all_sounds = deque(filteredkeys)
        splits = []
        cols = args.ratio
        size = args.size
        used_families = []
        samples = []
        total_samples = {}

        # For each sample
        for col in cols:
            sound_count = (col/100) * size
            families_in_column = []
            sounds_in_column = []
            sample = {}
            # Populate the sample until it has reached the required size
            while len(sounds_in_column) < sound_count:
                next_sound = all_sounds.popleft()
                nextsoundfamily = parsed_metadata[next_sound]['instrument_str']
                # If a sound doesn't have variants in another sample, consider it for inclusion
                if nextsoundfamily not in used_families:
                    # Cap the number of variants in a sample
                    if families_in_column.count(nextsoundfamily) < args.cap:
                        families_in_column.append(nextsoundfamily)
                        sounds_in_column.append(next_sound)
                        sample[next_sound] = parsed_metadata[next_sound]
                        total_samples[next_sound] = parsed_metadata[next_sound]
                else:
                    all_sounds.append(next_sound)
            splits.append(sounds_in_column)
            used_families.extend(families_in_column)
            samples.append(sample)
        return samples, total_samples

def get_mode(args):
    """Find the mode value for an attribute in the metadata."""
    with open(args.metadata) as metadata:
        # Parse the metadata
        parsed_metadata = json.load(metadata)
        keylist = list(parsed_metadata.keys())

        # Find the most common value for an attribute
        attribute_frequency_distr = {}
        for key in keylist:
            attribute = parsed_metadata[key][args.attribute]
            if attribute in attribute_frequency_distr:
                attribute_frequency_distr[attribute] += 1
            else:
                attribute_frequency_distr[attribute] = 1
        return max(attribute_frequency_distr, key=attribute_frequency_distr.get)

def save_sample(dest, sample):
    """Save a sample to a file."""
    print("Saving sample of size %s to %s..." % (len(sample), dest))
    try:
        with open(dest, 'w') as output:
            json.dump(sample, output, indent = 2)
    # If the file didn't exist, tell the user
    except FileNotFoundError:
        print("Working file not found, initialising...")

if __name__ == "__main__":
    args = parser.parse_args()

    # Get the mode of a specified attribute
    print("Getting mode of %s..." % (args.attribute))
    mode = get_mode(args)

    # Create samples holding that attribute consistent
    print("Creating a set of samples from %s with %s %s..." % (args.metadata, args.attribute, mode))
    samples, total_samples = create_samples(args, mode)
    save_sample(args.total, total_samples)
    save_sample(args.train, samples[0])
    save_sample(args.valid, samples[1])
    save_sample(args.test, samples[2])
