import os
import typing
import numpy as np
import pandas as pd

from llfizz.llphyscore import GridScore
from llfizz.constants import DATA_DIRECTORY, feature_tagABs

__all__ = ["FeatureVector", "Featurizer"]

# TODO: Constantly returning FeatureVector objects is inefficient. Should we modify in place?


class FeatureVector:
    """
    A mapping of feature names to feature values.
    """

    def __init__(
        self,
        seqid: str,
        feature_names: typing.List[str],
        feature_values: typing.List[float],
    ) -> None:
        self.seqid = seqid

        dtype = [("name", "U20"), ("value", "f8")]

        feature_names = np.array(feature_names, dtype="U20")
        feature_values = np.array(feature_values, dtype="f8")

        if len(feature_names) != len(feature_values):
            raise ValueError(
                "feature_names and feature_values must have the same length"
            )

        self.features = np.zeros(len(feature_names), dtype=dtype)
        self.features["name"] = feature_names
        self.features["value"] = feature_values

    @staticmethod
    def load(input_file: str) -> typing.Iterator["FeatureVector"]:
        """
        Load a feature vector from a CSV file.

        Parameters
        ----------
        input_file : str
            The path to the input CSV file.
        """
        # TODO: Can we make this work? Should be faster, but not needed enough to spend time optimizing.
        # data = np.genfromtxt(input_file, delimiter=',', names=True, dtype=None)
        # return FeatureVector(data['seqid'], data['name'], data['value'])

        feature_vectors = []
        with open(input_file, "r") as f:
            header = f.readline().strip().split(",")  # Read and split the first row
            feature_names = header[1:]  # Exclude 'seqid' from the header

            for line in f:
                values = line.strip().split(",")
                seqid = values[0]  # First value in row
                feature_values = np.array(
                    values[1:], dtype=np.float32
                )  # Convert values to float

                feature_vectors.append(
                    FeatureVector(seqid, feature_names, feature_values)
                )

        return feature_vectors

    @staticmethod
    def dump(feature_vectors: typing.List["FeatureVector"], output_file: str) -> None:
        """
        Dump the feature vector to a CSV file.

        Parameters
        ----------
        feature_vectors : List[FeatureVector]
            The list of feature vectors to dump.
        output_file : str
            The path to the output CSV file.
        """
        # TODO: Is text writing more efficient than a csv.writer?
        header_written = False
        with open(output_file, "w") as f:
            for fv in feature_vectors:
                if not header_written:
                    f.write("seqid," + ",".join(fv.features["name"]) + "\n")
                    header_written = True

                f.write(
                    f"{fv.seqid}," + ",".join(fv.features["value"].astype(str)) + "\n"
                )

    def concat(self, other: "FeatureVector") -> "FeatureVector":
        """
        Concatenate two feature vectors.

        Parameters
        ----------
        other : FeatureVector
            The other feature vector to concatenate.
        """
        if self.seqid != other.seqid:
            raise ValueError("FeatureVectors must have the same seqid to concatenate")

        # TODO: There's *no* way this the most efficient way of doing this...oh well ;)
        return FeatureVector(
            self.seqid,
            list(self.features["name"]) + list(other.features["name"]),
            list(self.features["value"]) + list(other.features["value"]),
        )

    def __eq__(self, other: "FeatureVector") -> bool:
        """
        Check if two feature vectors are equal.

        Parameters
        ----------
        other : FeatureVector
            The other feature vector to compare.
        """
        return (
            np.array_equal(self.features, other.features) and self.seqid == other.seqid
        )

    def __sub__(self, other: "FeatureVector") -> "FeatureVector":
        """
        Subtract two feature vectors.

        Parameters
        ----------
        other : FeatureVector
            The other feature vector to subtract.
        """
        return FeatureVector(
            self.seqid,
            self.features["name"],
            self.features["value"] - other.features["value"],
        )

    def __mul__(self, other: "FeatureVector") -> "FeatureVector":
        """
        Multiply two feature vectors.

        Parameters
        ----------
        other : FeatureVector
            The other feature vector to multiply.
        """
        return FeatureVector(
            self.seqid,
            self.features["name"],
            self.features["value"] * other.features["value"],
        )

    def square(self) -> "FeatureVector":
        """
        Square the feature vector.
        """
        return FeatureVector(
            self.seqid, self.features["name"], np.square(self.features["value"])
        )

    def get_feature_values(self):
        """
        Return the feature values.
        """
        return self.features["value"]

    def map_values(self, func) -> "FeatureVector":
        """Applies `func` to values of the feature vector."""
        func = np.vectorize(func)
        return FeatureVector(
            self.seqid, self.features["name"], func(self.features["value"])
        )

    # TODO: Add cmv functions.


class Featurizer:
    def __init__(self, funcs: typing.Dict[str, typing.Callable[..., float]]) -> None:
        """`funcs` is a dictionary mapping feature names to (sequence -> float) functions."""
        self._funcs = funcs

    def featurize(
        self,
        sequence: typing.Tuple,
        *,
        acceptable_errors=(ArithmeticError, ValueError, KeyError),
    ) -> typing.Tuple["FeatureVector", typing.Dict[str, Exception]]:
        """Compute the feature vector of a single sequence, and also return its failed computations."""
        feature_names = []
        feature_values = []
        errors = {}

        for featname, func in self._funcs.items():
            try:
                feature_values.append(func(sequence[1]))
                feature_names.append(featname)
            except acceptable_errors as e:
                errors[featname] = e
        return FeatureVector(sequence[0], feature_names, feature_values), errors

    # TODO: LLPhyScore doesn't have individual funcs that can be loaded in a custom config, which renders `featurize` method unhelpful. Need to think more about best design here.
    def llphyscore_featurize(
        self, sequence: typing.Tuple, score_db_dir: str
    ) -> typing.Tuple["FeatureVector", typing.Dict[str, Exception]]:
        """
        Compute the feature vector of a single sequence using LLPhyscore, and also return its failed computations.
        """
        seqid, seq = sequence

        feature_names = []
        feature_values = []
        errors = (
            {}
        )  # TODO: Empty because all error handling done in internal LLPhyscore code.

        for feature in feature_tagABs:
            # load GridScore database for one feature
            tagA, tagB = feature_tagABs[feature][0], feature_tagABs[feature][1]
            feature_grid_score = GridScore(
                dbpath=score_db_dir + "/{}".format(feature),
                tagA=tagA,
                tagB=tagB,
                max_xmer=40,
            )

            # generate the one-feature grids for all seqs.
            # TODO: Make this compatible with featurizing multiple sequences, ie. for seqid, seq in sequences.items()
            _tag, res_scores = feature_grid_score.score_sequence((seqid, seq))
            feature_grid = {tagA: [], tagB: []}

            for r in res_scores:
                feature_grid[tagA].append(r.A)
                feature_grid[tagB].append(r.B)

            feature_names += [tagA, tagB]
            feature_values += [np.mean(feature_grid[tagA]), np.mean(feature_grid[tagB])]

        return FeatureVector(seqid, feature_names, feature_values), errors

    # TODO: Make this function compatible with changes to featurize method
    def featurize_to_matrices(
        self,
        sequences: typing.Iterable[typing.Tuple[typing.Any, str]],
        *,
        acceptable_errors=(ArithmeticError, ValueError, KeyError),
    ) -> typing.Tuple[
        typing.Dict[typing.Any, "FeatureVector"],
        typing.Dict[typing.Any, typing.Dict[str, Exception]],
    ]:
        """
        Compute the feature vector of many sequences, and also return their failed computations.

        The two returned dicts (feature vector and errors) are indexed by whatever `sequences` was indexed by.
        """
        fvecs_all = {}
        errors_all = {}
        for label, sequence in sequences:
            fvec, errors = self.featurize(sequence, acceptable_errors=acceptable_errors)
            fvecs_all[label] = fvec
            if errors:
                errors_all[label] = errors
        return fvecs_all, errors_all
