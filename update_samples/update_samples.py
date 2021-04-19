#!/usr/bin/env python

"""Update training, validation and test .json files with feature and evaluation data.
The inputs are three metadata .json files, and a collated .json with additional data.
The output overwrites the three metadata files with updated versions of them."""

import numpy as np
import argparse
import json
import os


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-new', dest='new', action='store', type=str, required=True, help="Path for new metadata")
parser.add_argument('-train', dest='train', action='store', type=str, required=True, help="Path for training sample metadata")
parser.add_argument('-valid', dest='valid', action='store', type=str, required=True, help="Path for validation sample metadata")
parser.add_argument('-test', dest='test', action='store', type=str, required=True, help="Path for testing sample metadata")

def update_sample(new, sample, train=0):
    """Update the metadata of a sample."""
    print("Updating %s..." % sample)

    # Parse the new metadata
    with open(new) as newmetadata:
        parsed_new = json.load(newmetadata)
        new_keylist = list(parsed_new.keys())

    # Parse the sample metadata
    with open(sample) as sample_metadata:
        parsed_sample = json.load(sample_metadata)
        sample_keylist = list(parsed_sample.keys())

    for samplekey in sample_keylist:
        # Replace entries in the sample keylist with more recent versions in the new keylist
        if samplekey in new_keylist:
            parsed_sample[samplekey] = parsed_new[samplekey]

        # If the sample is training data
        if train:
            for newkey in new_keylist:
                # Add augmented versions of entries to the sample keylist
                if 'source' in parsed_new[newkey]:
                    if parsed_new[newkey]['source'] == samplekey:
                        parsed_sample[newkey] = parsed_new[newkey]
                        # Add evaluation data to the augmented versions
                        parsed_sample[newkey]['evaluations'] = parsed_sample[samplekey]['evaluations']
    return parsed_sample

def save_sample(dst, sample):
    """Save a sample to a file."""
    print("Saving updates to %s..." % dst)
    with open(dst, 'w') as output:
        json.dump(sample, output, indent = 2)

if __name__ == "__main__":
    args = parser.parse_args()

    # Update training metadata
    newtrain = update_sample(args.new, args.train, train=1)
    save_sample(args.train, newtrain)

    # Update validation metadata
    newvalid = update_sample(args.new, args.valid)
    save_sample(args.valid, newvalid)

    # Update testing metadata
    newtest = update_sample(args.new, args.test)
    save_sample(args.test, newtest)

    # Parse the new metadata
    with open(args.new) as newmetadata:
        parsed_new = json.load(newmetadata)
        new_keylist = list(parsed_new.keys())
