import typing
import numpy as np
from llfizz.featurizer import FeatureVector


class Metric:
    """Object holding an `origin` and a `weights` vector."""

    def __init__(self, origin: FeatureVector, weights: FeatureVector) -> None:
        self.origin = origin
        self.weights = weights

    def euclidean_norm_of(self, fvec: FeatureVector) -> float:
        """Compute the Euclidean distance from a feature vector to `self.origin`"""
        return self.euclidean_distance_between(fvec, self.origin)

    def euclidean_distance_between(
        self, fvec_a: FeatureVector, fvec_b: FeatureVector
    ) -> float:
        """Compute the Euclidean distance between two feature vectors."""
        # TODO: Make this more efficient?
        # TODO: This is senstive to small numerical errors from saving and reopening file. Worth fixing?
        z = (fvec_a - fvec_b) * self.weights
        norm = np.sum(z.square().get_feature_values())
        return norm

    @staticmethod
    def load(input_file: str) -> "Metric":
        """
        Load an origin and weight feature vector from a csv.

        Example
        -------
        metric = Metric.load("origin.csv, "weight.csv")
        """
        feature_vectors = FeatureVector.load(input_file)
        origin, weights = feature_vectors[0], feature_vectors[1]
        return Metric(origin, weights)

    def dump(self, output_file: str) -> None:
        """
        Write out the origin and weights feature vector to a csv.
        """
        FeatureVector.dump([self.origin, self.weights], output_file)
