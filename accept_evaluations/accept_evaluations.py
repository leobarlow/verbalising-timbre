#!/usr/bin/env python

"""Accept metaphorical evaluations for audio files, one scale at a time.
The inputs are audio files and a metadata .json for them.
The output is a metadata .json with evaluations added to it.
The program plays the audio files in a random order, and for each file
waits for the user to input an evaluation as an integer between 0 and 9,
which is written to the ouptut .json."""

import simpleaudio as sa
import argparse
import random
import json
import sys
import os


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-metadata', dest='metadata', action='store', type=str, required=True, help="Path for audio metadata")
parser.add_argument('-audio', dest='audio', action='store', type=str, required=True, help="Path for audio files")
parser.add_argument('-output', dest='output', action='store', type=str, required=True, help="Path for metadata with evaluations")
parser.add_argument('-aspect', dest='aspect', action='store', type=str, required=True, help="Timbral aspect to be evaluated")
parser.add_argument('--sample', dest='sample', action='store', type=int, required=False, help="A percentage of the timbres to reevaluate")

def is_aspect_valid(evaluation):
    """Check aspect value is numeric value between 0 and 9."""
    try:
        return (0 <= int(evaluation) <= 9)
    except:
        return False

def save_evaluations(eval_metadata, output):
    """Save updates to the metadata."""
    print("Saving evaluations to %s..." % output)
    try:
        with open(output, 'w') as eval_path:
            json.dump(eval_metadata, eval_path, indent = 2)
    # If the file didn't exist, tell the user
    except FileNotFoundError:
        print("Working file not found, initialising...")

def play_sound(key, audio):
    """Play an audio file."""
    soundpath = os.path.join(audio, key + '.wav')
    wave_obj = sa.WaveObject.from_wave_file(soundpath)
    play_obj = wave_obj.play()
    return

def get_evaluations(args, stop_command, repeat_command):
    """Accept evaluations for the audio data."""
    # Parse the metadata
    with open(args.metadata) as metadata:
        parsed_metadata = json.load(metadata)
        keylist = list(parsed_metadata.keys())

        # If a sample size is given, only take evaluations for that sample
        if args.sample:
            # Choose the sample with a random seed
            random.Random(1).shuffle(keylist)
            samplesize = int(len(keylist)/args.sample)
            keylist = keylist[0:samplesize]

        # Randomise the order of the sounds
        random.shuffle(keylist)

        # If there's already an output file, load it
        try:
            with open(args.output) as output:
                eval_metadata = json.load(output)
        # Else, initialise it by copying over the metadata
        except FileNotFoundError:
            print("Working file not found, initialising...")
            eval_metadata = parsed_metadata

        # Iterate through the audio metadata
        for i, key in enumerate(keylist):
            # If there isn't an evaluations entry in the metadata, create one
            if 'evaluations' not in eval_metadata[key]:
                eval_metadata[key]['evaluations'] = {}
            # If the aspect hasn't already been evaluated, evaluate it
            if args.aspect not in eval_metadata[key]['evaluations']:
                evaluation = '_'
                while not is_aspect_valid(evaluation):
                    play_sound(key, args.audio)
                    evaluation = input("Input evaluation for attribute %s of sound %s (%s of %s)\n"
                    "Alternatively press %s to repeat or %s to quit: " % (args.aspect, key, i + 1, len(keylist), repeat_command, stop_command))
                    if evaluation == stop_command:
                        return None
                    if not is_aspect_valid(evaluation) and evaluation != repeat_command:
                        print("Bad input: %s not in integer range 0-9" % evaluation)
                eval_metadata[key]['evaluations'][args.aspect] = int(evaluation)
                save_evaluations(eval_metadata, args.output)
        return None

if __name__ == "__main__":
    args = parser.parse_args()

    # Set the stop and repeat commands
    stop_command = 's'
    repeat_command = 'r'

    # Accept metaphorical evaluations
    get_evaluations(args, stop_command, repeat_command)
