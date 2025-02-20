import os
import typing
import numpy as np

from llfizz.llphyscore import GridScore
from llfizz.constants import DATA_DIRECTORY, feature_tagABs

__all__ = ['FeatureVector', 'Featurizer']

class FeatureVector:
    """
    A mapping of feature names to feature values.
    """
    def __init__(self, seqid: str, feature_names: typing.List[str], feature_values: typing.List[float]) -> None:
        self.seqid = seqid

        dtype = [('name', 'U20'), ('value', 'f8')]

        feature_names = np.array(feature_names, dtype='U20')
        feature_values = np.array(feature_values, dtype='f8')

        if len(feature_names) != len(feature_values):
            raise ValueError("feature_names and feature_values must have the same length")

        self.features = np.zeros(len(feature_names), dtype=dtype)
        self.features['name'] = feature_names
        self.features['value'] = feature_values
    
    def dump(self, output_file: str) -> None:
        """
        Dump the feature vector to a CSV file.
        
        Parameters
        ----------
        output_file : str
            The path to the output CSV file.
        """
        featvalues = np.insert(self.features['value'].astype(str), 0, self.seqid)
        np.savetxt(output_file, [featvalues], delimiter=',', header='seqid,'+','.join(self.features['name']), comments='', fmt='%s')

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
        return FeatureVector(self.seqid, list(self.features['name']) + list(other.features['name']), list(self.features['value']) + list(other.features['value']))

    # TODO: Add cmv, load, map_values functions.
 
class Featurizer:
    def __init__(self, funcs: typing.Dict[str, typing.Callable[...,float]]) -> None:
        """`funcs` is a dictionary mapping feature names to (sequence -> float) functions."""
        self._funcs = funcs

    def featurize(self, sequence: typing.Tuple, *, acceptable_errors = (ArithmeticError, ValueError, KeyError)) -> typing.Tuple["FeatureVector", typing.Dict[str, Exception]]:
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
    def llphyscore_featurize(self, sequence: typing.Tuple, score_db_dir: str) -> typing.Tuple["FeatureVector", typing.Dict[str, Exception]]:
        """
        Compute the feature vector of a single sequence using LLPhyscore, and also return its failed computations.
        """
        seqid, seq = sequence

        feature_names = []
        feature_values = []
        errors = {} # TODO: Empty because all error handling done in internal LLPhyscore code.

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
    def featurize_to_matrices(self, sequences: typing.Iterable[typing.Tuple[typing.Any, str]], *, acceptable_errors = (ArithmeticError, ValueError, KeyError)) -> typing.Tuple[typing.Dict[typing.Any, "FeatureVector"], typing.Dict[typing.Any, typing.Dict[str, Exception]]]:
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
