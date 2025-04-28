"""WIP: Script for designing feature-based mimics."""

import os
import json
import random
import sys
import csv
import argparse
import tqdm
import pathos
import tqdm_pathos
import multiprocessing
from contextlib import nullcontext

from llfizz.metrics import Metric
from llfizz.designer import FeatureDesigner
from llfizz.featurizer import Featurizer
from llfizz.features import compile_native_featurizer
from llfizz.constants import AMINOACIDS, MAX_RETRIES
from benchstuff import Fasta

LENGTH_THRESHOLD = 30
SEED_COLNAME = "Seed"
MAX_SEED = 2**64
CONVERGENCE_THRESHOLD = 1e-4
GOOD_MOVES_THRESHOLD = 3


def init_subprocess(stderr_lock, output_lock):
    global STDERR_LOCK
    STDERR_LOCK = stderr_lock
    global OUTPUT_LOCK
    OUTPUT_LOCK = output_lock


def parse_args():
    parser = argparse.ArgumentParser(
        "feature-mimic", description="design feature mimics of the given IDR regions"
    )

    inputs = parser.add_argument_group("design-inputs")
    inputs.add_argument(
        "input_sequences", help="whole protein sequences in fasta format"
    )
    inputs.add_argument(
        "feature_weights_file", help="features csv containing a weight feature vector"
    )
    parser.add_argument(
        "output_file",
        help="output csv file with columns (`ProteinID`, `DesignID`, `Sequence`, ...)",
    )
    parser.add_argument(
        "feature_set",
        help="feature set to use for featurizing sequences. Options: 'original', 'hybrid', 'LLPhyScore'.",
        default="original",
    )

    rng_seed = parser.add_mutually_exclusive_group()
    rng_seed.add_argument(
        "--n-random",
        type=int,
        help="sample this many random query sequences per region",
    )

    parser.add_argument(
        "--feature-file",
        required=False,
        help="feature configuration json",
        default="data/feature_config/native-features.json",
    )
    parser.add_argument(
        "--keep-trajectory",
        action="store_true",
        help="when set, save every iteration of the design loop",
    )
    parser.add_argument(
        "--save-seed",
        action="store_true",
        help="when set, store the seed in a `Seed` column",
    )
    parser.add_argument(
        "--design-id",
        required=False,
        default="{counter}",
        help="string to format the design id. default is to use a counter",
    )
    parser.add_argument(
        "-np",
        "--n-processes",
        type=int,
        required=False,
        default=1,
        help="number of processes. requires libraries: pathos + tqdm_pathos",
    )

    return parser.parse_args()


def design_task(
    query,
    target,
    feature_set,
    protid,
    regionid,
    designid,
    seed,
    designer,
    colnames,
    args,
    acceptable_errors=(ArithmeticError, ValueError, KeyError),
):
    featurizer = Featurizer(designer.featurizer_func_dict)

    designer.metric.origin, _ = featurizer.featurize("target", target, feature_set)

    designer.rng.seed(seed)

    if query is None:
        for _ in range(MAX_RETRIES):
            try_query = "".join(
                designer.rng.choice(AMINOACIDS) for _ in range(len(target))
            )

            try:
                featurizer.featurize("try_query", try_query, feature_set)
            except acceptable_errors:
                continue

            query = try_query
            break

    try:
        if args.keep_trajectory:
            save = []
            for progress in designer.design_loop(
                query, feature_set, acceptable_errors=acceptable_errors
            ):
                save.append(progress)
        else:
            for progress in designer.design_loop(
                query, feature_set, acceptable_errors=acceptable_errors
            ):
                progress.pop("Iteration")
                save = [progress]
    except acceptable_errors:
        with STDERR_LOCK:
            print(
                "query did not have all features (protid=%s,regionid=%s,seed=%d)"
                % (protid, regionid, seed),
                file=sys.stderr,
            )

    with OUTPUT_LOCK:
        with open(args.output_file, "a") as file:
            writer = csv.DictWriter(file, colnames)
            for row in save:
                row["ProteinID"] = protid
                row["DesignID"] = designid
                if regionid is not None:
                    row["RegionID"] = regionid
                if args.save_seed:
                    row["Seed"] = seed
                writer.writerow(row)


def design_all(num_processes, tasks):
    if num_processes > 1:
        stderr_lock = multiprocessing.Lock()
        output_lock = multiprocessing.Lock()
        pool = pathos.multiprocessing.Pool(
            num_processes,
            initializer=init_subprocess,
            initargs=(stderr_lock, output_lock),
        )
        with pool:
            tqdm_pathos.map(lambda task: design_task(*task), tasks, pool=pool)
    else:
        global STDERR_LOCK
        STDERR_LOCK = nullcontext()
        global OUTPUT_LOCK
        OUTPUT_LOCK = nullcontext()
        for task in tqdm.tqdm(tasks, desc="designing sequences"):
            design_task(*task)


def main():
    args = parse_args()

    metric = Metric.load(args.feature_weights_file)

    if args.feature_file is not None:
        with open(args.feature_file, "r") as file:
            features_dict = json.load(file)
    featurizer, errors = compile_native_featurizer(args.feature_set, features_dict)

    for featname, error in errors.items():
        print("error compiling `%s`: %s" % (featname, error), file=sys.stderr)

    if set(featurizer.keys()) != set(metric.weights.features["name"]):
        print(set(featurizer.keys()).difference(set(metric.weights.features["name"])))
        print(set(metric.weights.features["name"]).difference(set(featurizer.keys())))
        raise RuntimeError(
            "featurizer and metric feature vector have different features"
        )

    designer = FeatureDesigner(
        featurizer,
        metric,
        covergence_threshold=CONVERGENCE_THRESHOLD,
        good_moves_threshold=GOOD_MOVES_THRESHOLD,
        rng=random.Random(),
    )

    fa = Fasta.load(args.input_sequences)
    Fasta.assume_unique = True

    tasks = []
    colnames = ["ProteinID", "DesignID", "Time", "Sequence"]
    featnames = featurizer.keys()

    if args.save_seed:
        colnames.append("Seed")

    if args.keep_trajectory:
        colnames.append("Iteration")

    colnames += featnames

    fa = fa.filter(lambda _, seq: len(seq) >= LENGTH_THRESHOLD)

    n_random = args.n_random or 1
    rng = random.Random()
    seeds = fa.to_protein_dict().map_values(
        lambda _: [rng.randint(0, MAX_SEED) for _ in range(n_random)]
    )

    for protid, prot_seeds in seeds:
        if (entry := fa.get(protid)) is None:
            continue
        _, target = entry
        assert isinstance(target, str)
        for counter, seed in enumerate(prot_seeds):
            seed = int(seed)
            design_id = args.design_id.format(
                counter=counter, seed=seed, proteinid=protid
            )
            tasks.append(
                (
                    None,
                    target,
                    args.feature_set,
                    protid,
                    None,
                    design_id,
                    seed,
                    designer,
                    colnames,
                    args,
                )
            )

    if not os.path.exists(args.output_file):
        with open(args.output_file, "w") as file:
            csv.DictWriter(file, colnames).writeheader()
        design_all(args.n_processes, tasks)
        return

    with open(args.output_file, "r") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            with open(args.output_file, "w") as file:
                csv.DictWriter(file, colnames).writeheader()
            design_all(args.n_processes, tasks)
            return

        if reader.fieldnames != colnames:
            BOLD_RED = "\033[1;31m"
            NORMAL = "\033[0m"
            print(
                BOLD_RED
                + "cannot overwrite file `%s` with different column names (shown below):"
                % args.output_file
                + NORMAL,
                file=sys.stderr,
            )
            print(",".join(colnames), file=sys.stderr)
            sys.exit(1)

        if args.keep_trajectory:
            checkpoint = [row for row in reader if row.pop("Iteration") == "END"]
        else:
            checkpoint = list(reader)

    checkpoint_keys = ["ProteinID", "DesignID"]
    keys = [2, 4]

    checkpoint = {
        tuple(row.pop(key) for key in checkpoint_keys): row for row in checkpoint
    }
    tasks_not_done = []
    for task in tasks:
        checkpoint_key = tuple(task[key] for key in keys)
        if checkpoint_key not in checkpoint:
            tasks_not_done.append(task)

    design_all(args.n_processes, tasks_not_done)


if __name__ == "__main__":
    main()
