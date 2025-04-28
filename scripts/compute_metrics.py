"""
Script for computing the mean and variance of features over a fasta file.

Example
-------
`$ python compute-metric.py input-sequences.fasta output.csv`
`$ python compute-metric.py input-features.csv output.csv`
"""

import math
import argparse

from llfizz.featurizer import FeatureVector
from llfizz.metrics import Metric


def parse_args():
    parser = argparse.ArgumentParser(
        "compute-metric",
        description="compute the mean and weights of a given feature vector spreadsheet",
    )
    parser.add_argument("input_file", help="input features csv file")
    parser.add_argument("output_file", help="output features csv file")
    return parser.parse_args()


def main():
    args = parse_args()
    feature_vectors = FeatureVector.load(args.input_file)
    mean, variance = FeatureVector.get_mean_and_var(feature_vectors)
    weights = variance.map_values(lambda x: 1 / math.sqrt(x) if x > 0 else 0)
    Metric(mean, weights).dump(args.output_file)


if __name__ == "__main__":
    main()
