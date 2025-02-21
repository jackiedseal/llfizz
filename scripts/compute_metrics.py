"""
Script for computing the mean and variance of features over a fasta file.

Example
-------
`$ python compute-metric.py input-sequences.fasta output.csv`
`$ python compute-metric.py input-features.csv output.csv`
"""

import json
import sys
import math
import tqdm
import argparse

from llfizz.featurizer import FeatureVector, Featurizer
from llfizz.features import compile_native_featurizer
from llfizz.metrics import Metric


def parse_args():
    parser = argparse.ArgumentParser(
        "compute-metric",
        description="compute the mean and weights of a given feature vector spreadsheet",
    )
    parser.add_argument("input_file", help="input features csv file")
    parser.add_argument("output_file", help="output features csv file")

    # TODO: Do we need these?
    # parser.add_argument("--input-labels", nargs="*", required=False, help="column name(s) containing identifying columns in the INPUT file")
    # parser.add_argument("--output-label", required=False, default="Label", help="column labelling the origin or weights vector in the OUTPUT file")
    # parser.add_argument("--feature-file", required=False, help="feature configuration json")
    return parser.parse_args()


def main():
    args = parse_args()

    featurizer, errors = compile_native_featurizer()
    for featname, error in errors.items():
        print("error compiling `%s`: %s" % (featname, error), file=sys.stderr)
    featurizer = Featurizer(featurizer)

    feature_vectors = FeatureVector.load(args.input_file)
    mean, variance = FeatureVector.get_mean_and_var(feature_vectors)
    weights = variance.map_values(lambda x: 1 / math.sqrt(x) if x > 0 else 0)
    Metric(mean, weights).dump(args.output_file)


if __name__ == "__main__":
    main()
