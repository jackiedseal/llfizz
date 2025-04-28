"""
Script for computing proteome feature vectors.

Example
-------
    $ python featurize.py input.fasta output.csv
"""

import argparse
import logging
from typing import Any, Dict, Tuple, List, Optional

import json
import tqdm

# Assuming benchstuff is installed and importable
from benchstuff import Fasta
from llfizz.featurizer import FeatureVector, Featurizer
from llfizz.features import compile_native_featurizer


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="featurize", description="Compute the feature vectors of a FASTA file."
    )
    parser.add_argument("input_sequences", help="Input FASTA file.")
    parser.add_argument("output_file", help="Output feature file (CSV format).")
    parser.add_argument(
        "feature_set",
        help="Feature set to use for featurizing sequences. Options: 'original', 'hybrid', 'LLPhyScore'.",
        default="original",
    )
    parser.add_argument(
        "--feature-file",
        help="Feature configuration JSON file. Used when feature_set is 'original' or 'hybrid'.",
        default=None,
    )
    return parser.parse_args()


def setup_logging() -> None:
    """
    Configure the logging format and level.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def compile_configured_featurizer(
    feature_set: str,
    feature_file: Optional[str],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Compile the featurizer either from a configuration file or use the native version.

    Parameters
    ----------
    feature_file : Optional[str]
        Path to a JSON file containing feature configuration. If None, use native featurizer.

    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        A tuple containing the compiled featurizer and any errors encountered during compilation.
    """
    if feature_file:
        with open(feature_file, "r") as file:
            config = json.load(file)
        featurizer_config, errors = compile_native_featurizer(feature_set, config)
    else:
        featurizer_config, errors = compile_native_featurizer(feature_set)
    return featurizer_config, errors


def report_errors(errors: Dict[Any, Any], context: str = "compiling") -> None:
    """
    Log errors from featurizer compilation or featurization process.

    Parameters
    ----------
    errors : Dict[Any, Any]
        Dictionary of errors.
    context : str, optional
        A string describing the context of the errors, by default "compiling".
    """
    for key, error in errors.items():
        logging.error(f"Error {context} `{key}`: {error}")


def featurize_sequences(
    featurizer: Featurizer,
    sequences: List[Tuple[Any, str]],
    output_file: str,
    feature_set: str,
) -> None:
    """
    Featurize a list of sequences and dump the feature vectors.

    Parameters
    ----------
    featurizer : Featurizer
        The featurizer to use.
    sequences : List[Tuple[Any, str]]
        A list of tuples where the first element is the identifier(s) and the second is the sequence.
    output_file : str
        The path to the output CSV file.
    feature_set : str
        The feature set to use for featurizing sequences. Options: 'original', 'hybrid', 'LLPhyScore'.
    """
    feature_vectors = []
    for sequence in tqdm.tqdm(
        sequences, total=len(sequences), desc="Featurizing sequences"
    ):
        # Note: Min length of sequence[1] is 2 to avoid errors with LLPhyScore computation.
        if len(sequence[1]) > 2:
            feature_vector, errors = featurizer.featurize(
                sequence[0], sequence[1], feature_set
            )
            report_errors(errors, context="featurizing")
            feature_vectors.append(feature_vector)

    FeatureVector.dump(feature_vectors, output_file)


def main() -> None:
    setup_logging()
    args = parse_args()

    # Compile the featurizer
    featurizer_config, compile_errors = compile_configured_featurizer(
        args.feature_set, args.feature_file
    )
    report_errors(compile_errors, context="compiling")
    featurizer = Featurizer(featurizer_config)

    # Load the sequences to be featurized
    fa = Fasta.load(args.input_sequences)

    seen = set()
    sequences = [
        (protein_id, seq)
        for protein_id, seq in fa
        if seq not in seen and not seen.add(seq)
    ]

    featurize_sequences(featurizer, sequences, args.output_file, args.feature_set)


if __name__ == "__main__":
    main()
