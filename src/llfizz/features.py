"""
Module of default/native feature functions.

See `compile_native_featurizer` to create sequence feature functions from a config
like `native-features.json` in the same directory.

See `compile_native_feature` for the format of configuration.
"""

import re
import typing

import os
import json

import numpy as np
from functools import partial

from llfizz.constants import DATA_DIRECTORY, feature_tagABs

__all__ = [
    "compile_native_featurizer",
    "compile_native_feature",
    "score_pattern_matches",
    "count_pattern_matches",
    "pattern_match_span",
    "log_ratio",
    "scd",
    "simple_spacing_closure",
    "custom_kappa_closure",
    "complexity",
    "isoelectric_point",
]


def compile_native_featurizer(features_dict=None):
    """
    Given a nested-dict object like `native-features.json` in this directory,
    make a dictionary of names to feature functions.

    If no dictionary is provided, `native-features.json` is used.

    Layout
    ------
    The `features_dict` has the following expected sections:
    {
        "features": { ... },
        "residue_groups": { ... },
        "motif_frequencies": { ... },
        "aa_frequencies": { ... }
    }

    The "features" section is a dict of (featname, feature json) pairs. The
    features json contains all the kwargs necessary to compile one sequence function,
    according to the `compile_native_feature` function just below this function.

    The "residue_groups" section is a dict of names to strings/lists, depicting certain
    special residue groups. (e.g. "charged" residues might be represented as {"charged": "DEKR"})
    These residue groups are required for computing `simple_spacing` and `percent_res_group`
    features (see `compile_native_feature`).

    The "motif_frequencies" section is an (optional) dict of motifs to their expected count per length.
    See `count_pattern_matches_minus_expected`.

    The "aa_frequencies" section is a dict of amino acids to their expected frequencies in the IDRome.
    See `repeats_minus_expected` and `compile_native_feature`.
    """
    if features_dict is None:
        with open(os.path.join(DATA_DIRECTORY, "native-features.json"), "r") as file:
            features_dict = json.load(file)
    features = features_dict["features"]
    residue_groups = features_dict.get("residue_groups") or {}
    motif_frequencies = features_dict.get("motif_frequencies") or {}
    residue_frequencies = features_dict.get("aa_frequencies")
    return_value = {}
    errors = {}
    for featname, feature_params in features.items():
        try:
            return_value[featname] = compile_native_feature(
                residue_groups=residue_groups,
                motif_frequencies=motif_frequencies,
                residue_frequencies=residue_frequencies,
                **feature_params,
            )
        except (ValueError, TypeError) as e:
            errors[featname] = e

    # TODO: Toggle this based on featurization strat.
    for featname in feature_tagABs:
        tagA, tagB = feature_tagABs[featname][0], feature_tagABs[featname][1]
        return_value[tagA] = np.nan
        return_value[tagB] = np.nan

    return return_value, errors


def compile_native_feature(
    *,
    compute: str,
    residue_groups: typing.Dict[str, str],
    motif_frequencies: typing.Dict[str, float],
    residue_frequencies: typing.Optional[typing.Dict[str, float]],
    **kwargs,
):
    """
    Turn the provided kwargs into one sequence to feature function.

    Parameters
    ----------
    The `residue_groups`, `motif_frequencies`, and `residue_frequencies`
    are broadly described in `compile_native_featurizer` and required for
    certain features below.

    The `compute` parameter determines what type of feature will be
    computed. The available ones are:
    - "score"
    - "count"
    - "percent_residue"
    - "percent_res_group"
    - "span"
    - "repeats"
    - "log_ratio"
    - "scd"
    - "simple_spacing"
    - "custom_kappa"
    - "complexity" | "sequence_complexity"
    - "isoelectric_point"

    The necessary keyword arguments by computation type are:

    `compute="score"`
    score : dict[str, float]
        A dictionary with numeric scores (values) for each motif/residue (keys).
    take_average | average : bool
        If true, divide the summed score by sequence length.

    `compute="count"`
    pattern : str
        The regex pattern to count.
    take_average | average : bool
        If true, divide the pattern count by sequence length.
    subtract_expected : bool
        If true, search the `motif_frequencies` for the expected count / length of this motif.
        Will raise if this value is true but no such expected value can be found.
        Uses the expected frequency in the `count_pattern_matches_minus_expected` function.

    `compute="percent_residue"`
    residue : str
        A single amino acid to get the composition of.

    `compute="percent_res_group"`
    residue_group : str
        The name of a residue group in the provided `residue_groups` dict.
        Raises if such a group is not present.

    `compute="span"`
    pattern : str
        The regex pattern to get the span of.

    `compute="repeats"`
    residues : str | list[str]
        The residues considered part of this class of repeats.
        (e.g. "FYW" would make this an aromatic repeat computation)
    subtract_expected : bool
        If true, uses `aa_frequencies` (which must not be None) to produce an
        expected number of repeat occurrences for each length.
        See `repeats_minus_expected`.

    `compute="log_ratio"`
    numerator : str
        Single amino acid. Positively correlated with feature value.
    denominator : str
        Single amino acid. Negatively correlated with feature value.

    `compute="simple_spacing"`
    residue_group : str
        The name of a residue group in the provided `residue_groups` dict.
        Raises if such a group is not present.

    `compute=other`
    No additional parameters required.
    """
    if compute == "score":
        if (score := kwargs.get("score")) is None:
            raise ValueError(
                "`compute=score` requires a `score` parameter (see `score_pattern_matches`)"
            )
        average = kwargs.get("take_average") or kwargs.get("average") or False
        if not isinstance(average, bool):
            raise TypeError("expected `average` to be True or False")
        if not isinstance(score, dict):
            raise TypeError("expected `score` to be a dict of residue->score pairs")
        return partial(score_pattern_matches, score=score, average=average)

    if compute == "count":
        if (pattern := kwargs.get("pattern")) is None:
            raise ValueError(
                "`compute=count` requires a `pattern` parameter (see `count_pattern_matches`)"
            )
        average = kwargs.get("take_average") or kwargs.get("average") or False
        if not isinstance(pattern, str):
            raise TypeError("expected `pattern` to be a regex pattern")
        if not isinstance(average, bool):
            raise TypeError("expected `average` to be True or False")
        if kwargs.get("subtract_expected"):
            if (pattern_frequency := motif_frequencies.get(pattern)) is None:
                raise ValueError(
                    "`subtract_expected` was set but no motif frequency for `%s` is available"
                    % pattern
                )
            if not isinstance(pattern_frequency, float):
                raise TypeError(
                    "expected motif frequency for `%s` to be a number" % pattern
                )
            return partial(
                count_pattern_matches_minus_expected,
                pattern=re.compile(pattern),
                pattern_frequency=pattern_frequency,
                average=average,
            )
        return partial(
            count_pattern_matches, pattern=re.compile(pattern), average=average
        )

    if compute == "percent_residue":
        if (residue := kwargs.get("residue")) is None:
            raise ValueError(
                "`compute=percent_residue` requires a residue as the `residue` parameter"
            )
        if not isinstance(residue, str) and len(residue) == 1:
            raise TypeError("expected `residue` to be a single amino acid")
        return partial(count_pattern_matches, pattern=re.compile(residue), average=True)

    if compute == "percent_res_group":
        if (res_group_name := kwargs.get("residue_group")) is None:
            raise ValueError(
                "`compute=percent_res_group` requires the name of a residue group as the `residue_group` parameter"
            )
        if (res_group := residue_groups.get(res_group_name)) is None:
            raise ValueError(
                "unknown residue group %s - available are: %s"
                % (res_group_name, ",".join(residue_groups.keys()))
            )
        if isinstance(res_group, list):
            res_group = "".join(res_group)
        if not isinstance(res_group, str):
            raise TypeError(
                "expected residue group %s to be a list or string of amino acids"
                % res_group_name
            )
        return partial(
            count_pattern_matches, pattern=re.compile("[%s]" % res_group), average=True
        )

    if compute == "span":
        if (pattern := kwargs.get("pattern")) is None:
            raise ValueError(
                "`compute=span` requires a `pattern` parameter (see `pattern_match_span`)"
            )
        return partial(pattern_match_span, pattern=re.compile(pattern))

    if compute == "repeats":
        if (residues := kwargs.get("residues")) is None:
            raise ValueError(
                "`compute=repeats` requires the residues in the repeat as the `residues` parameter"
            )
        if isinstance(residues, list):
            residues = "".join(residues)
        if not isinstance(residues, str):
            raise TypeError("expected `residues` to be a list or string of amino acids")
        if kwargs.get("subtract_expected"):
            if residue_frequencies is None:
                raise ValueError(
                    "`subtract_expected` was set but no residue frequencies are available"
                )
            residue_frequency = sum(residue_frequencies.get(aa, 0) for aa in residues)
            return partial(
                repeats_minus_expected,
                repeat_pattern=re.compile("[%s]" % residues),
                residue_frequency=residue_frequency,
            )
        return partial(pattern_match_span, pattern=re.compile("[%s]" % residues))

    if compute == "log_ratio":
        if (num_aa := kwargs.get("numerator")) is None:
            raise ValueError(
                "`compute=log_ratio` requires a `numerator` parameter (see `log_ratio`)"
            )
        if (denom_aa := kwargs.get("denominator")) is None:
            raise ValueError(
                "`compute=log_ratio` requires a `denominator` parameter (see `log_ratio`)"
            )
        return partial(log_ratio, num_aa=num_aa, denom_aa=denom_aa)

    if compute == "scd":
        return scd

    if compute == "simple_spacing":
        if (res_group_name := kwargs.get("residue_group")) is None:
            raise ValueError(
                "`compute=simple_spacing` requires the name of a residue group as the `residue_group` parameter"
            )
        if (res_group := residue_groups.get(res_group_name)) is None:
            raise ValueError(
                "unknown residue group %s - available are: %s"
                % (res_group_name, ",".join(residue_groups.keys()))
            )
        if isinstance(res_group, list):
            res_group = "".join(res_group)
        if not isinstance(res_group, str):
            raise TypeError(
                "expected residue group %s to be a list or string of amino acids"
                % res_group_name
            )
        return simple_spacing_closure(res_group, res_group_name)

    if compute == "custom_kappa":
        return custom_kappa_closure()

    if compute == "complexity" or compute == "sequence_complexity":
        return complexity

    if compute == "isoelectric_point":
        return isoelectric_point

    raise ValueError("not a recognized compute option: %s" % compute)


def score_pattern_matches(
    sequence: str,
    score: dict[re.Pattern[str], float],
    average: bool,
) -> float:
    """Calculate a weighted count or average of regex occurrences.

    Parameters
    ----------
    sequence : str
        Target sequence on which to perform the weighted count of
        regex occurrences.

    scores : dict[str, float]
        A dictionary containing the regex patterns to look for and the weights
        they contribute to the count.

    average : bool, optional
        Whether to divide by sequence length at the end.

    Raises
    ------
    If `average` is ``True`` and the provided sequence is empty.
    """
    result = sum(score * len(re.findall(pat, sequence)) for pat, score in score.items())
    if average:
        return result / len(sequence)
    return result


def count_pattern_matches(
    sequence: str,
    pattern: re.Pattern[str],
    average: bool,
) -> float:
    """Count or average the number of regex occurrences in a sequence.

    Parameters
    ----------
    sequence : str
        Target sequence on which to count the regex occurrences.

    pattern : str
        The regex pattern to count.

    average : bool, optional
        Whether to divide by sequence length at the end.

    Raises
    ------
    If `average` is ``True`` and the provided sequence is empty.
    """
    result = len(re.findall(pattern, sequence))
    if average:
        return result / len(sequence)
    return result


def count_pattern_matches_minus_expected(
    sequence: str,
    pattern: re.Pattern[str],
    pattern_frequency: float,
    average: bool,
) -> float:
    """Count or average the number of regex occurrences in a sequence.

    Parameters
    ----------
    sequence : str
        Target sequence on which to count the regex occurrences.

    pattern : str
        The regex pattern to count.

    pattern_frequency : float
        The expected occurrence-per-residue of this pattern.

    average : bool, optional
        Whether to divide by sequence length at the end.

        Defaults to ``False``.

    Raises
    ------
    If `average` is ``True`` and the provided sequence is empty.
    """
    result = len(re.findall(pattern, sequence))
    result -= pattern_frequency * (len(sequence) - len(pattern.pattern))
    if average:
        return result / len(sequence)
    return result


def pattern_match_span(
    sequence: str,
    pattern: re.Pattern[str],
) -> float:
    """Calculate the total length spanned by patterns in a target sequence.

    Parameters
    ----------
    sequence : str
        Target sequence on which to determine the spanning length of the regex
        occurrences.

    pattern : str
        The regex pattern to determine the length of.
    """
    return sum(
        right - left
        for left, right in map(re.Match.span, re.finditer(pattern, sequence))
    )


def repeats_minus_expected(
    sequence: str, repeat_pattern: re.Pattern[str], residue_frequency: float
) -> float:
    """Calculate the total length spanned by patterns in a target sequence.

    Parameters
    ----------
    sequence : str
        Target sequence on which to determine the spanning length of the regex
        occurrences.

    repeat_pattern : str
        The repeat as a regex pattern to determine the length of.

    residue_frequency : float
        The total probability of a uniformly drawn amino acid of being considered
        as part of the repeat residues. (e.g. for K/R repeats, it would be p(K) + p(R))

    Expected number
    ---------------
    Given a residue_frequency P describing the probability of a
    residue in [`Repeats::residues`] occuring by chance, the
    probability that any residue is part of a repeat sequence is:

    ```
        Case 2
        v
    XXXXXXXXXXX
    ^         ^
    Case 1    Case 3
    ```

    1. The residue and its right neighbour must both be in the
        residue set. This occurs with probability P^2.
    2. The residue and its left or right neighbour must be in the
        residue set. This occurs with probability 2 P^2. However,
        we've double counted the case where the residue and the
        left and right neighbour are in the set, so we subtract
        P^3.
    3. Residue and its left neighbour must be in residue set.
        See 1.

    ```
    The resulting expression is like this:
    2 P^2 + (L-2) x (2 P^2 - P^3)
    -----    -------------------
    edges           middle

    Which goes to
    P^2 (2 x (L-1) - (L-2) x P)
    ```
    """
    expected_span = (
        residue_frequency
        * residue_frequency
        * ((2 * (len(sequence) - 1)) - ((len(sequence) - 2) * residue_frequency))
    )
    return (
        sum(
            right - left
            for left, right in map(re.Match.span, re.finditer(repeat_pattern, sequence))
        )
        - expected_span
    )


def log_ratio(
    sequence: str,
    num_aa: str,
    denom_aa: str,
) -> float:
    from math import log

    """Calculate ``log(1 + num_aa) - log(1 - denom_aa)`` for in a sequence.

    Uses natural log.

    Parameters
    ----------
    sequence : str
        Target sequence on which to determine the log ratio.

    num_aa : str
        The amino acid in the numerator.

    denom_aa : str
        The amino acid in the denominator.
    """
    return log((1 + sequence.count(num_aa)) / (1 + sequence.count(denom_aa)))


def scd(sequence: str) -> float:
    """
    Calculate the `SCD`_ (Sequence Charge Decoration) of a sequence.

    .. _SCD: https://doi.org/10.1063/1.5005821

    Parameters
    ----------
    sequence : str
        Target sequence on which to determine the SCD.
    """
    from math import sqrt

    BINARY_CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1}
    charged_res = []
    for i, aa in enumerate(sequence):
        if aa in BINARY_CHARGE:
            charged_res.append(i)
    if not charged_res:
        return 0
    result: float = 0
    for i, loc_i in enumerate(charged_res):
        for loc_j in charged_res[:i]:
            result += (
                BINARY_CHARGE[sequence[loc_i]]
                * BINARY_CHARGE[sequence[loc_j]]
                * sqrt(loc_i - loc_j)
            )
    return result / len(sequence)


def abstract_spacing_calculation(
    *,
    sequence: str,
    candidates: typing.List[int],
    are_neighbours: typing.Callable[[str, str], bool],
    prob_neighbor_given_candidate: typing.Callable[[str, typing.List[int], int], float],
    not_enough_candidates_error: str,
    blob: int,
) -> float:
    """Calculate a spacing parameter.

    This is an abstraction over two similar computations: `simple_spacing`
    and `custom_kappa`. They both involve counting two subsets of residues:
    1. `candidates`, given by the provided list, and
    2. `neighbours`, which are a subset of those candidates that must be
        within close proximity to each other.

    The resulting number is a measure of how often close-neighbour residue
    pairs occur compared to how often they are expected to occur via
    composition.

    This function is not publically exposed as a `*` import.
    """
    from math import sqrt

    def count_neighbors():
        """The number of `neighbour` residues."""
        candidate_pairs = list(zip(candidates[:-1], candidates[1:]))
        if not candidate_pairs:
            raise ValueError(not_enough_candidates_error)
        return sum(
            1
            for i, j in candidate_pairs
            if are_neighbours(sequence[i], sequence[j]) and (abs(i - j) <= blob)
        )

    actual_neighbors = count_neighbors()
    p: float = prob_neighbor_given_candidate(sequence, candidates, blob)
    mean_neighbors: float = p * len(candidates)
    sd_neighbors: float = sqrt(p * (1 - p) * len(candidates))
    return (actual_neighbors - mean_neighbors) / sd_neighbors


def simple_spacing_closure(
    res_group: str,
    res_group_name: str,
    *,
    blob: int = 5,
):
    """
    Set up a closure to compute simple spacing for this residue group.

    In a simple spacing computation, the residues in a certain residue group (e.g.
    the charged residues, the aromatic residues, etc.) are considered to be
    `candidate` residues. The pairs of candidate residues within 5 residues of each
    other are the `neighbour` residues.

    This closure therefore measures how "clustered" the residues from this residue
    group are distributed in the sequence.
    """

    def prob_neighbor_given_candidate(
        sequence: str,
        candidates: typing.List[int],
        blob: int,
    ):
        proportion_candidates: float = len(candidates) / len(sequence)
        # It shouldn't be zero either, but that should be caught already.
        if proportion_candidates == 1:
            raise ValueError(
                "cannot compute %s spacing on sequence with only %s residues"
                % (res_group_name, res_group_name)
            )
        return proportion_candidates * sum(
            (1 - proportion_candidates) ** i for i in range(blob)
        )

    def are_neighbours(_a: str, _b: str):
        return True

    not_enough_candidates_error = "sequence has no %s residues" % res_group_name

    def simple_spacing(sequence: str):
        candidates = []
        for i, aa in enumerate(sequence):
            if aa in res_group:
                candidates.append(i)
        return abstract_spacing_calculation(
            sequence=sequence,
            candidates=candidates,
            are_neighbours=are_neighbours,
            prob_neighbor_given_candidate=prob_neighbor_given_candidate,
            not_enough_candidates_error=not_enough_candidates_error,
            blob=blob,
        )

    return simple_spacing


def custom_kappa_closure(
    *,
    blob: int = 5,
):
    """
    Set up a closure to compute a kappa-like measure for this residue group.

    In this custom-kappa computation, charged residues are candidates, and the pairs of
    same-charge residues within 5 residues of each other are the `neighbour` residues.

    This closure therefore measures how similar/blocky charges residues are.
    """
    BINARY_CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1}

    def prob_neighbor_given_candidate(
        sequence: str,
        candidates: typing.List[int],
        blob: int,
    ):
        proportion_candidates: float = len(candidates) / len(sequence)
        count_pos: int = sum(1 for i in candidates if BINARY_CHARGE[sequence[i]] == 1)
        count_neg: int = len(candidates) - count_pos
        prob_charges_are_diff: float = (
            2 * (count_pos) * (count_neg) / (len(candidates) ** 2)
        )
        if proportion_candidates == 1 and prob_charges_are_diff == 0:
            raise ValueError(
                "cannot compute charge spacing, `custom_kappa` cannot deal with all residues of one charge"
            )
        prob_next_charge_in_blob: float = proportion_candidates * sum(
            (1 - proportion_candidates) ** i for i in range(blob)
        )
        return prob_next_charge_in_blob * (1 - prob_charges_are_diff)

    def are_neighbours(res_a: str, res_b: str):
        return BINARY_CHARGE[res_a] == BINARY_CHARGE[res_b]

    not_enough_candidates_error = "sequence has no charged residues"

    def custom_kappa(sequence: str):
        candidates = []
        for i, aa in enumerate(sequence):
            if aa in BINARY_CHARGE:
                candidates.append(i)
        return abstract_spacing_calculation(
            sequence=sequence,
            candidates=candidates,
            are_neighbours=are_neighbours,
            prob_neighbor_given_candidate=prob_neighbor_given_candidate,
            not_enough_candidates_error=not_enough_candidates_error,
            blob=blob,
        )

    return custom_kappa


def complexity(sequence: str) -> float:
    """Calculate the `complexity`_ (entropy-like feature) of a target sequence.

    Uses natural log.

    .. _complexity: https://www.sciencedirect.com/science/article/pii/009784859385006X

    Parameters
    ----------
    sequence : str
        Target sequence on which to determine the complexity.

    Raises
    ------
    If the sequence is empty.
    """  # noqa: E501, pylint: disable=line-too-long
    from math import lgamma

    AMINOACIDS = "ACDEFGHIKLMNPQRSTVWY"
    log_gamma_sum: float = 0
    for aa in AMINOACIDS:
        log_gamma_sum += lgamma(1 + sequence.count(aa))
    return (lgamma(1 + len(sequence)) - log_gamma_sum) / len(sequence)


def accurate_net_charge(
    ph: float,
    num_basic_res: int,
    counts_and_pkas: typing.Iterable[tuple[int, float]],
) -> float:
    """Calculate a very accurate net charge based on pKa formulas.

    Helper for :func:`~isoelectric_point`.

    Parameters
    ----------
    ph : float
        The pH to be calculated at.

    num_basic_res: int
        The number of sites which are positively charged when protonated,
        including the basic N terminus site. i.e.
        0 is not a valid value because there is always a basic N terminus site.

    counts_and_pkas : Iterable[tuple[int, float]]
        The number of sites of a specific pKa, for each pKa.

    Raises
    ------
    If an invalid (non-positive) `num_basic_res` is passed in.

    Note
    ----
    **Math logic**

    The logic behind this function is that one counts the basic sites
    (positively charged in their protonated state) as the default charge.

    As you increase the pH, a proportion of those sites become deprotonated,
    decreasing the charge. The proportion of deprotonated sites of species
    with a fixed pka is:

    >>> 1 / (1 + 10 ** (pka - ph))

    So you subtract the expected number of free protons (corresponding to
    deprotonated sites) from the number of basic sites.
    """

    PKA_N_TERM = 7.5
    PKA_C_TERM = 3.55

    if num_basic_res < 1:
        raise AssertionError(
            "Trying to calculate net charge on "
            + "a negative number of basic residues! (%d)" % num_basic_res,
        )
    free_protons: float = 0
    for count, pka in counts_and_pkas:
        proportion_protonated = 1 / (1 + 10 ** (ph - pka))
        free_protons += count * (1 - proportion_protonated)
    proportion_protonated = 1 / (1 + (10 ** (ph - PKA_N_TERM)))
    free_protons += 1 - proportion_protonated
    proportion_protonated = 1 / (1 + (10 ** (ph - PKA_C_TERM)))
    free_protons += 1 - proportion_protonated
    return num_basic_res - free_protons


def binary_search_root_finder(
    f: typing.Callable[[float], float],
    bracket: tuple[float, float],
    threshold: float = 1e-4,
) -> float:
    """Find roots of a decreasing sigmoidal function.

    Dependency for `isoelectric_point`.

    Parameters
    ----------
    f : Callable[[float], float]
        Function to find the root of. Should only have one root.

    guess : float, optional
        Initial guess.

    bracket : tuple[float, float]
        Interval on which to find the root.
    """
    bottom, top = bracket
    guess = (top + bottom) / 2
    while top - bottom > threshold:
        if f(guess) > 0:
            bottom = guess
        else:
            top = guess
        guess = (top + bottom) / 2
    return guess


def isoelectric_point(sequence: str):
    """Calculate the isoelectric point of a target sequence.

    Searches for a root of the pH-charge curve on the pH interval 0 to 14.

    Parameters
    ----------
    sequence : str
        Target sequence on which to determine the isoelectric point.

    Raises
    ------
    If the charge curve is negative at the acidic end or positive at the
    basic end, it is guaranteed not to have a root on the interval.
    """
    from functools import partial

    PKAS_ALL = {
        "D": 4.05,
        "E": 4.45,
        "C": 9.0,
        "Y": 10.0,
        "K": 10.0,
        "R": 12.0,
        "H": 5.98,
    }
    BASIC_RES = "KHR"

    counts_and_pkas: list[tuple[int, float]] = [
        (sequence.count(aa), pka) for aa, pka in PKAS_ALL.items()
    ]
    num_basic_res: int = 1 + sum(1 for aa in sequence if aa in BASIC_RES)

    continuous_charge: typing.Callable[[float], float] = partial(
        accurate_net_charge,
        num_basic_res=num_basic_res,
        counts_and_pkas=counts_and_pkas,
    )
    if (continuous_charge(0) < 0) or (continuous_charge(14) > 0):
        raise ValueError(
            "Isoelectric point of the protein " + "cannot be on the interval [0, 14]."
        )
    return binary_search_root_finder(continuous_charge, (0, 14))
