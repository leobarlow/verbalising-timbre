#!/usr/bin/env python

"""Generate plots and calculate summary statistics for metaphorical evaluation data.
The inputs are a metadata .json for the evaluated dataset, and an optional .json
for a reevaluated sample of it.
The outputs are Spearman's rank correlation coefficients for the evaluated scales,
the normality of the evaluated scale distributions, and their means and standard
deviations. If a reevaluated sample is given, the program also outputs Tau-equivalent
reliability between the evaluations in the sample and the full evaluation data.
It produces heatmap plots for the scale correlations, and bar graphs for the scale
distributions."""

from matplotlib import pyplot as plt
from collections import Counter
import tikzplotlib as tpl
from scipy import stats
import numpy as np
import argparse
import shutil
import json
import os


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-att1', dest='att1', action='store', type=str, required=True, help="Attribute 1")
parser.add_argument('-att2', dest='att2', action='store', type=str, required=True, help="Attribute 2")
parser.add_argument('-att3', dest='att3', action='store', type=str, required=True, help="Attribute 3")
parser.add_argument('-total', dest='total', action='store', type=str, required=True, help="Path for total metadata")
parser.add_argument('--sample', dest='sample', action='store', type=str, required=False, help="Path for reevaluated sample metadata")

def reliability(eval_pairs):
    """Get the tau-equivalent reliability coefficient for two sets of evaluations."""
    itemvars = eval_pairs.var(axis=1, ddof=1)
    tscores = eval_pairs.sum(axis=0)
    nitems = len(eval_pairs)
    return nitems / (nitems-1.) * (1 - itemvars.sum() / tscores.var(ddof=1))

def get_reliabilities(args):
    """Return the reliability of the total and reevaluated samples."""
    # Parse the total metadata
    with open(args.total) as total:
        parsed_total = json.load(total)
        totalkeys = list(parsed_total.keys())

    # Parse the sample metadata
    with open(args.sample) as sample:
        parsed_sample = json.load(sample)
        samplekeys = list(parsed_sample.keys())

    # Store original and reevaluated values for attribute 1
    att1pairs = []
    att1 = []
    # Get all the reevaluated values
    for samplekey in [key for key in samplekeys if 'evaluations' in parsed_sample[key]]:
        totalvalue = parsed_total[samplekey]['evaluations'][args.att1]
        samplevalue = parsed_sample[samplekey]['evaluations'][args.att1]
        att1pairs.append([totalvalue, samplevalue])
        att1.append(totalvalue)
    att1pairs = np.swapaxes(np.array(att1pairs), 0, 1)
    att1reliability = reliability(att1pairs)

    # Store original and reevaluated values for attribute 2
    att2pairs = []
    att2 = []
    # Get all the reevaluated values
    for samplekey in [key for key in samplekeys if 'evaluations' in parsed_sample[key]]:
        totalvalue = parsed_total[samplekey]['evaluations'][args.att2]
        samplevalue = parsed_sample[samplekey]['evaluations'][args.att2]
        att2pairs.append([totalvalue, samplevalue])
        att2.append(totalvalue)
    att2pairs = np.swapaxes(np.array(att2pairs), 0, 1)
    att2reliability = reliability(att2pairs)

    # Store original and reevaluated values for attribute 3
    att3pairs = []
    att3 = []
    # Get all the reevaluated values
    for samplekey in [key for key in samplekeys if 'evaluations' in parsed_sample[key]]:
        totalvalue = parsed_total[samplekey]['evaluations'][args.att3]
        samplevalue = parsed_sample[samplekey]['evaluations'][args.att3]
        att3pairs.append([totalvalue, samplevalue])
        att3.append(totalvalue)
    att3pairs = np.swapaxes(np.array(att3pairs), 0, 1)
    att3reliability = reliability(att3pairs)
    return att1reliability, att2reliability, att3reliability

def get_correlations(args, att1, att2, att3):
    """Return correlations between attribute pairs."""
    # Return Spearman's correlation for each attribute pair
    spearman12, p12 = stats.spearmanr(att1, att2)
    spearman13, p13 = stats.spearmanr(att1, att3)
    spearman23, p23 = stats.spearmanr(att2, att3)
    return (spearman12, p12), (spearman13, p13), (spearman23, p23)

def plot_bars(args, att1, att2, att3):
    """Plot attribute distributions as bar graphs."""
    # Get the frequency of the highest-frequency value
    counts = []
    counts.append(np.max(np.unique(att1, return_counts=True)[1]))
    counts.append(np.max(np.unique(att2, return_counts=True)[1]))
    counts.append(np.max(np.unique(att3, return_counts=True)[1]))
    max = np.max(counts)

    att1counts = list(zip(*sorted(Counter(att1).items())))[1]
    att2counts = list(zip(*sorted(Counter(att2).items())))[1]
    att3counts = list(zip(*sorted(Counter(att3).items())))[1]

    # Plot the evaluations as bar graphs
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ticks = range(10)
    ax1.bar(range(10), att1counts, width=1, edgecolor='black', color='grey', linewidth=0.1, align='center')
    ax1.set_xticks(ticks)
    ax2.bar(range(10), att2counts, width=1, edgecolor='black', color='grey', linewidth=0.1, align='center')
    ax2.set_xticks(ticks)
    ax3.bar(range(10), att3counts, width=1, edgecolor='black', color='grey', linewidth=0.1, align='center')
    ax3.set_xticks(ticks)

    # Set the frequencies to one scale
    space = max + 7
    ax1.set_ylim([0, space])
    ax2.set_ylim([0, space])
    ax3.set_ylim([0, space])
    ax1.set_xlabel(args.att1.capitalize())
    ax1.set_ylabel('Frequency')
    ax2.set_xlabel(args.att2.capitalize())
    ax3.set_xlabel(args.att3.capitalize())
    return fig

def plot_heatmaps(args, att1, att2, att3):
    """Plot attribute pairs as heatmaps."""
    # Find the highest frequency of pairs
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    a = ax1.hist2d(att1, att2)
    b = ax2.hist2d(att1, att3)
    c = ax3.hist2d(att2, att3)
    max = np.max(np.array((a[0], b[0], c[0])))
    fig.clear()

    # Plot the correlations as heat maps
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    axlist = [ax1, ax2, ax3]

    # Set the plots to one scale
    a = ax1.hist2d(att1, att2, bins=10, cmap='Greys', vmin=0, vmax=max)
    b = ax2.hist2d(att1, att3, bins=10, cmap='Greys', vmin=0, vmax=max)
    c = ax3.hist2d(att2, att3, bins=10, cmap='Greys', vmin=0, vmax=max)
    fig.colorbar(c[3], ax=axlist, label='Frequency')
    ax1.set_xlabel(args.att1.capitalize())
    ax1.set_ylabel(args.att2.capitalize())
    ax2.set_xlabel(args.att1.capitalize())
    ax2.set_ylabel(args.att3.capitalize())
    ax3.set_xlabel(args.att2.capitalize())
    ax3.set_ylabel(args.att3.capitalize())
    return fig

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

    # Parse the total metadata
    with open(args.total) as total:
        parsed_total = json.load(total)
        totalkeys = list(parsed_total.keys())

    # Store original evaluations values
    att1 = []
    att2 = []
    att3 = []
    for key in totalkeys:
        att1.append(parsed_total[key]['evaluations'][args.att1])
        att2.append(parsed_total[key]['evaluations'][args.att2])
        att3.append(parsed_total[key]['evaluations'][args.att3])

    # Plot bar graphs of each attribute distribution
    plot_bars(args, att1, att2, att3)
    save_fig('distributions')

    # Print the mean and standard deviation of each attribute distribution
    print("Mean and standard deviation of {} = {:.3f}, {:.3f}".format(args.att1, np.mean(att1), np.std(att1)))
    print("Mean and standard deviation of {} = {:.3f}, {:.3f}".format(args.att2, np.mean(att2), np.std(att2)))
    print("Mean and standard deviation of {} = {:.3f}, {:.3f}".format(args.att3, np.mean(att3), np.std(att3)))

    # Print tau-equivalent reliabilities for each attribute
    if args.sample:
        att1reliability, att2reliability, att3reliability = get_reliabilities(args)
        print("Attribute {} reliability = {:.3f}".format(args.att1, att1reliability))
        print("Attribute {} reliability = {:.3f}".format(args.att2, att2reliability))
        print("Attribute {} reliability = {:.3f}".format(args.att3, att3reliability))

    # Print normality coefficients for each attribute
    print("Shapiro-wilk normality coefficient for {} = {:.3f} (p = {})".format(args.att1, stats.shapiro(att1)[0], stats.shapiro(att1)[1]))
    print("Shapiro-wilk normality coefficient for {} = {:.3f} (p = {})".format(args.att2, stats.shapiro(att2)[0], stats.shapiro(att2)[1]))
    print("Shapiro-wilk normality coefficient for {} = {:.3f} (p = {})".format(args.att3, stats.shapiro(att3)[0], stats.shapiro(att3)[1]))

    # Print Spearman's correlation for each attribute pair
    att1att2, att1att3, att2att3 = get_correlations(args, att1, att2, att3)
    print("Spearman's correlation for {} and {} = {:.3f} (p = {})".format(args.att1, args.att2, att1att2[0], att1att2[1]))
    print("Spearman's correlation for {} and {} = {:.3f} (p = {})".format(args.att1, args.att3, att1att3[0], att1att3[1]))
    print("Spearman's correlation for {} and {} = {:.3f} (p = {})".format(args.att2, args.att3, att2att3[0], att2att3[1]))

    # Plot heatmaps of each attribute pair
    plot_heatmaps(args, att1, att2, att3)
    save_fig('heatmaps')
