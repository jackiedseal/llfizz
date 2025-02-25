import typing
import numpy as np

from llfizz.llphyscore import GridScore
from llfizz.constants import DATA_DIRECTORY, SCORE_DB_DIR, feature_tagABs

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

    # TODO: Wait, we don't need this method. We can just use the `features` attribute directly.
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

    @staticmethod
    def get_mean_and_var(feature_vectors: typing.List["FeatureVector"]) -> typing.Tuple["FeatureVector", "FeatureVector", "FeatureVector"]:
        """
        Returns a mean and variance  tuple from the iterable of feature vectors.

        Variance is population variance (i.e. uses `N` instead of `N-1` in the denominator.)
        """
        assert len(feature_vectors) > 0, "No feature vectors to compute cmv from."

        featnames = feature_vectors[0].features["name"]
        mean_feature_vector = FeatureVector("mean", featnames, np.nanmean([fv.features["value"] for fv in feature_vectors], axis=0))
        var_feature_vector = FeatureVector("variance", featnames, np.nanvar([fv.features["value"] for fv in feature_vectors], axis=0))

        return mean_feature_vector, var_feature_vector



class Featurizer:
    def __init__(self, funcs: typing.Dict[str, typing.Callable[..., float]]) -> None:
        """`funcs` is a dictionary mapping feature names to (sequence -> float) functions."""
        self._funcs = funcs
        self.grid_score_cache = {}

    def get_funcs(self) -> typing.List[str]:
        """Return the functions used by the featurizer."""
        return self._funcs.values()
    
    def featurize(
        self,
        seqid: str,
        sequence: str,
        *,
        acceptable_errors=(ArithmeticError, ValueError, KeyError),
    )  -> typing.Tuple["FeatureVector", typing.Dict[str, Exception]]:
        native_feature_vector, native_errors = self.vanilla_featurize(seqid, sequence)
        llphys_feature_vector, llphys_errors = self.llphyscore_featurize(seqid, sequence, SCORE_DB_DIR)
        feature_vector = native_feature_vector.concat(llphys_feature_vector)
        return feature_vector, native_errors | llphys_errors
       
    def vanilla_featurize(
        self,
        seqid: str,
        sequence: str,
        *,
        acceptable_errors=(ArithmeticError, ValueError, KeyError),
    ) -> typing.Tuple["FeatureVector", typing.Dict[str, Exception]]:
        """Compute the feature vector of a single sequence, and also return its failed computations."""
        feature_names = []
        feature_values = []
        errors = {}

        for featname, func in self._funcs.items():
            if func is not np.nan:
                feature_names.append(featname)
                try:
                    feature_values.append(func(sequence))
                except acceptable_errors as e:
                    feature_values.append(np.nan)
                    errors[featname] = e

        return FeatureVector(seqid, feature_names, feature_values), errors

    # TODO: LLPhyScore doesn't have individual funcs that can be loaded in a custom config, which renders `featurize` method unhelpful. Need to think more about best design here.
    def llphyscore_featurize(
        self, seqid: str, sequence: str, score_db_dir: str
    ) -> typing.Tuple["FeatureVector", typing.Dict[str, Exception]]:
        """
        Compute the feature vector of a single sequence using LLPhyscore, and also return its failed computations.
        """
        feature_names = []
        feature_values = []
        errors = (
            {}
        )  # TODO: Empty because all error handling done in internal LLPhyscore code.

        for feature in feature_tagABs:
            # load GridScore database for one feature
            tagA, tagB = feature_tagABs[feature][0], feature_tagABs[feature][1]

            if feature not in self.grid_score_cache:
                self.grid_score_cache[feature] = GridScore(
                    dbpath=score_db_dir + "/{}".format(feature),
                    tagA=tagA,
                    tagB=tagB,
                    max_xmer=40,
                )

            feature_grid_score = self.grid_score_cache[feature]

            # generate the one-feature grids for all seqs.
            # TODO: Make this compatible with featurizing multiple sequences, ie. for seqid, seq in sequences.items()
            _tag, res_scores = feature_grid_score.score_sequence((seqid, sequence))
            feature_grid = {tagA: [], tagB: []}

            for r in res_scores:
                feature_grid[tagA].append(r.A)
                feature_grid[tagB].append(r.B)

            feature_names += [tagA, tagB]

            self._funcs[tagA] = np.nan
            self._funcs[tagB] = np.nan

            feature_values += [np.mean(feature_grid[tagA]), np.mean(feature_grid[tagB])]

        return FeatureVector(seqid, feature_names, feature_values), errors

    # TODO: Do we need a featurize_to_matrices method? 
