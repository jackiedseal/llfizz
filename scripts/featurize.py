"""
Script for computing proteome feature vectors.

Example
-------
    $ python featurize.py input.fasta output.csv
"""

import argparse
import logging
from typing import Any, Dict, Tuple, List, Optional

import tqdm
import numpy as np

# Assuming benchstuff and idrfeatlib are installed and importable
from benchstuff import Fasta, Regions
from llfizz.featurizer import FeatureVector, Featurizer
from llfizz.features import compile_native_featurizer
from llfizz.constants import SCORE_DB_DIR
from llfizz.metrics import Metric


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
    parser.add_argument(
        "--input-regions",
        help="Input regions CSV. Format: ProteinID,RegionID,Start,Stop",
        default=None,
    )
    parser.add_argument("output_file", help="Output feature file (CSV format).")
    parser.add_argument(
        "--feature-file", help="Feature configuration JSON file.", default=None
    )
    return parser.parse_args()


def setup_logging() -> None:
    """
    Configure the logging format and level.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def compile_configured_featurizer(
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
    # if feature_file:
    #     with open(feature_file, "r") as file:
    #         config = json.load(file)
    #     featurizer_config, errors = compile_featurizer(config)
    # else:
    #     featurizer_config, errors = compile_native_featurizer()
    featurizer_config, errors = compile_native_featurizer()
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
    output_file: str,  # <-- Explicitly pass output_file
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
    """
    # TODO: This will need to handle multiple sequences at some point.
    native_feat_vector, errors = featurizer.featurize(sequences[0])
    llphys_feat_vector, errors = featurizer.llphyscore_featurize(
        sequences[0], SCORE_DB_DIR
    )
    feat_vector = native_feat_vector.concat(llphys_feat_vector)
    report_errors(errors, context="featurizing")

    FeatureVector.dump([feat_vector], output_file)


def main() -> None:
    setup_logging()
    args = parse_args()

    # Compile the featurizer
    featurizer_config, compile_errors = compile_configured_featurizer(args.feature_file)
    report_errors(compile_errors, context="compiling")

    featurizer = Featurizer(featurizer_config)
    fa = Fasta.load(args.input_sequences)

    sequences = [
        (protein_id, seq)
        for protein_id, seq in tqdm.tqdm(
            fa, total=len(fa), desc="Featurizing sequences"
        )
    ]
    featurize_sequences(
        featurizer, sequences, args.output_file
    )  # <-- Pass args.output_file

    # TODO: Implement region-based featurization.
    # if args.input_regions is None:
    #     sequences = [(protein_id, seq) for protein_id, seq in tqdm.tqdm(fa, total=len(fa), desc="Featurizing sequences")]
    #     featurize_sequences(featurizer, sequences, ("ProteinID",), args.output_file)  # <-- Pass args.output_file
    # else:
    #     regions, _ = Regions.load(args.input_regions)
    #     Fasta.assume_unique = True
    #     sequences: List[Tuple[Tuple[str, str], str]] = []

    #     for prot_id, region_id, (start, stop) in regions.iter_nested():
    #         entry = fa.get(prot_id)
    #         if entry is None:
    #             logging.warning(f"Protein `{prot_id}` not found in FASTA; skipping region `{region_id}`.")
    #             continue

    #         _, whole_seq = entry
    #         if not isinstance(whole_seq, str):
    #             logging.error(f"Expected sequence string for protein `{prot_id}`, got {type(whole_seq)} instead.")
    #             continue

    #         seq = whole_seq[start:stop]
    #         if len(seq) != stop - start:
    #             logging.error(
    #                 f"Invalid region `{region_id}` for protein `{prot_id}` "
    #                 f"(start={start}, stop={stop}, seqlen={len(whole_seq)})"
    #             )
    #             continue
    #         sequences.append(((prot_id, region_id), seq))

    #     featurize_sequences(featurizer, sequences, ("ProteinID", "RegionID"), args.output_file)  # <-- Pass args.output_file


if __name__ == "__main__":
    main()
