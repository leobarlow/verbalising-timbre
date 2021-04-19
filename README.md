## Verbalising Timbre: A Neural Network Approach

This repository contains the code and judgement data I created to predict metaphorical timbre judgements using deep neural networks.

### Data

The audio files can be downloaded from the [NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth), which I collated locally into `data\all\audio` and a metadata file called `examples.json`.

* `eval_total.json` is the metadata file with my original judgements for *brightness*, *softness* and *fullness*

* `sample_eval_total.json` is the metadata file with my second set of judgements

### Data collection

The inputs to the data collection pipeline are the `.wav` files in `audio` and the metadata in `examples.json`.

1. Use `create_samples.py` to create training, validation and test sets from `examples.json`. The outputs are to `init_total.json`, `init_train.json`, `init_valid.json` and `init_test.json`.

2. Use `reduce_audio.py` to normalise and crop the audio files referenced by `init_total.json`. The output is to `reduced_audio`.

3. Use `accept_evaluations.py` to add timbre judgements (or 'evaluations') for the audio files referenced by `init_total.json`, which are stored in `eval_total.json`.

4. Use `augment_data.py` to augment the audio files referenced by `init_total.json`. The audio output is to `augmented_audio` and the updated metadata is stored in `aug_eval_total.json`.

5. Use `extract_features.py` to extract features from the audio files in `augmented_audio`, and add them to the metadata in `aug_eval_total.json`.

7. Use `update_samples.py` to copy feature and evaluation data from `aug_eval_total.json` to `init_train.json`, `init_valid.json` and `init_test.json`. This also updates `init_train.json` with the augmented metadata entries.

### Data analysis

* `check_evaluations.py` runs some statistical tests on the evaluation data, and produces various plots.

### Neural network modelling

1. Use `train_model.py` to build, compile and train neural a neural network design. The version with the lowest loss is saved to `trained_model.h5`.

2. Use `test_model.py` to test the neural network, which outputs plots and accuracy metrics.
