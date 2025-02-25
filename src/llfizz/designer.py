import typing
from copy import copy
from time import time
from math import sqrt
from random import Random

from llfizz.features import scd
from llfizz.metrics import Metric
from llfizz.featurizer import Featurizer
from llfizz.constants import AMINOACIDS, N_AA, CHARGE

__all__ = [
    "FeatureDesigner",
]

class FeatureDesigner:
    """Feature mimic design algorithm."""

    # TODO: Why is `featurizer` a dictionary of callables? It should be a Featurizer object.
    def __init__(
        self,
        featurizer: typing.Dict[str, typing.Callable[..., float]],
        metric: Metric,
        covergence_threshold: float,
        good_moves_threshold: int,
        rng: Random,
    ) -> None:
        self.featurizer_func_dict = featurizer
        self.metric = metric
        self.convergence_threshold = covergence_threshold
        self.good_moves_threshold = good_moves_threshold
        self.rng = rng  # TODO: "rng" is not a good name for a (presumably) random number generator. 

    def generate_mutations(self, sequence_length: int):
        """Helper for `design_loop`."""
        seeds = list(range(sequence_length * N_AA))
        self.rng.shuffle(seeds)
        for seed in seeds:
            aa = AMINOACIDS[seed % N_AA]
            pos = int(seed / N_AA)
            yield pos, aa

    def design_loop(
        self,
        query_sequence: str,
        acceptable_errors=(ArithmeticError, ValueError, KeyError),
    ):
        """
        Optimize a query sequence to fit the feature vector at `self.metric.origin`.

        Parameters
        ----------
        query_sequence : str
            The starting sequence to generate designs off of by iterative mutation.
        """
        # TODO: tuple vs string input?
        query_fvec, _ = Featurizer(self.featurizer_func_dict).vanilla_featurize(
            'query_sequence', query_sequence, acceptable_errors=()
        )

        scd_key = None
        scd_machine = None

        if scd in self.featurizer_func_dict.values():
            if scd_machine is None:
                previous_scd = query_fvec.features[query_fvec.features["name"] == "scd"]["value"].item()

                charged_res = set()
                for i, aa in enumerate(query_sequence):
                    if aa in CHARGE:
                        charged_res.add(i)

                mutation_from = ""
                mutation_pos = len(query_sequence)

                scd_machine = ScdMachine(previous_scd, charged_res, mutation_from, mutation_pos)
                scd_key = "scd"

            self.featurizer_func_dict["scd"] = scd_machine.compute_scd

        featurizer = Featurizer(self.featurizer_func_dict)

        start_time = time()
        iteration = 0

        yield {
            **dict(zip(query_fvec.features["name"], query_fvec.features["value"])),
            "Iteration": iteration,
            "Sequence": query_sequence,
            "Time": 0,
        }

        current_distance_to_target: float = self.metric.euclidean_norm_of(query_fvec)
        next_distance_to_target: float = current_distance_to_target
        current_distance_to_target += self.convergence_threshold + 1  # TODO: why do? specifically, the +1.

        while current_distance_to_target - next_distance_to_target > self.convergence_threshold:
            print(iteration)
            current_distance_to_target = next_distance_to_target
            all_candidates = [(query_sequence, copy(scd_machine))]  # TODO: huh? why copy scd_machine?
            best_machine = scd_machine

            for mutation_pos, mutation_to in self.find_lower_mutations(
                self.generate_mutations(len(query_sequence)),
                current_distance_to_target,
                query_sequence,
                featurizer,
                scd_machine,
                acceptable_errors=acceptable_errors,
            ):
                new_candidates = []
                for old_candidate, old_scd_machine in all_candidates:
                    mutation_from = old_candidate[mutation_pos]
                    if old_scd_machine is not None:
                        assert scd_machine is not None
                        scd_machine.clone_from(old_scd_machine, shallow=False)

                    assert (
                        candidate := apply_mutation(
                            old_candidate, mutation_pos, mutation_to, scd_machine
                        )
                    ) is not None

                    try:
                        candidate_fvec, _ = featurizer.vanilla_featurize(
                            'candidate', candidate, acceptable_errors=()
                        )
                    except acceptable_errors:
                        continue

                    candidate_distance_to_target = self.metric.euclidean_norm_of(candidate_fvec)

                    if scd_machine is not None:
                        assert scd_key is not None
                        new_scd = candidate_fvec.features[query_fvec.features["name"] == "scd"]["values"]
                        scd_machine.advance_mutation(mutation_pos, mutation_to, new_scd)

                        new_scd_machine = scd_machine.clone(shallow=True) #TODO: Why do we need a refresh on scd machine every time?
                        mutation_from = old_candidate[mutation_pos]
                    else:
                        new_scd_machine = None

                    if next_distance_to_target > candidate_distance_to_target:
                        next_distance_to_target = candidate_distance_to_target
                        query_fvec = candidate_fvec
                        query_sequence = candidate
                        best_machine = new_scd_machine

                    new_candidates.append((candidate, new_scd_machine))

                all_candidates.extend(new_candidates)
                if scd_machine is not None:
                    scd_machine.clone_from(best_machine, shallow=True)

            iteration += 1

            yield {
                **dict(zip(query_fvec.features["name"], query_fvec.features["value"])),
                "Iteration": iteration,
                "Sequence": query_sequence,
                "Time": time() - start_time,
            }

        yield {
            **dict(zip(query_fvec.features["name"], query_fvec.features["value"])),
            "Iteration": "END",
            "Sequence": query_sequence,
            "Time": time() - start_time,
        }

        return query_sequence # TODO: wait, this is indeed a tuple, no? should revert to calling it `query`?


    def find_lower_mutations(
        self,
        mutation_generator: typing.Iterator[typing.Tuple[int, str]],
        current_distance_to_target: float,
        current_sequence: str,
        featurizer: Featurizer,
        scd_machine: typing.Optional["ScdMachine"],
        acceptable_errors: ...,
    ) -> typing.List[typing.Tuple[int, str]]:
        """
        Find single point mutations that decrease the distance to a target.

        Bins them into those that decrease the distance to target a lot
        ("good_moves") and those that decrease the distance by a small amount
        ("decent_moves").
        """
        def add_mutation_to_memo(
            mutation_pos: int,
            mutation_to: str,
            distance_to_target: float,
            memo: dict[int, typing.Tuple[str, float]],
        ):
            """Helper for `find_lower_mutations`."""
            if (collision := memo.get(mutation_pos)) is None:
                memo[mutation_pos] = mutation_to, distance_to_target
                return
            _, memoized_dist = collision
            if distance_to_target >= memoized_dist:
                return
            memo[mutation_pos] = mutation_to, distance_to_target

        good_mutation_distances = {}
        decent_mutation_distances = {}

        for mutation_pos, mutation_to in mutation_generator:
            if (guess := apply_mutation(current_sequence, mutation_pos, mutation_to, scd_machine)) is None:
                continue

            try:
                guess_fvec, _ = featurizer.vanilla_featurize('guess', guess, acceptable_errors=())
            except acceptable_errors:
                continue

            guess_distance_to_target = self.metric.euclidean_norm_of(guess_fvec)
            if guess_distance_to_target < current_distance_to_target:
                if (current_distance_to_target - guess_distance_to_target > self.convergence_threshold):
                    add_mutation_to_memo(mutation_pos, mutation_to, guess_distance_to_target, good_mutation_distances)
                else:
                    add_mutation_to_memo(mutation_pos, mutation_to, guess_distance_to_target, decent_mutation_distances)

            if len(good_mutation_distances) >= self.good_moves_threshold:
                break

        mutations = {pos: aa for pos, (aa, _) in decent_mutation_distances.items()} 
        mutations.update({pos: aa for pos, (aa, _) in good_mutation_distances.items()})
        return list(mutations.items())


def apply_mutation(sequence: str, mutation_pos: int, mutation_to: str, scd_machine: typing.Optional["ScdMachine"]):
    """Apply a point mutation to a sequence, appropriately adjusting the `scd_machine`."""
    if (mutation_from := sequence[mutation_pos]) == mutation_to:
        return None
    if scd_machine is not None:
        scd_machine.mock_mutation(mutation_from, mutation_pos)
    return sequence[:mutation_pos] + mutation_to + sequence[mutation_pos+1:]


class ScdMachine:
    """A helper for the mimic design algorithm that computes SCD in non-quadratic time."""
    def __init__(self, previous_scd: float, charged_res: typing.Set[int], mutation_from: str, mutation_pos: int) -> None:
        self.previous_scd = previous_scd
        self.charged_res = charged_res
        self.mutation_from = mutation_from
        self.mutation_pos = mutation_pos

    def compute_scd(self, sequence: str):
        """Compute the SCD from a previous SCD and mutation data."""
        ch_delta = CHARGE.get(sequence[self.mutation_pos], 0) - CHARGE.get(self.mutation_from, 0)

        if ch_delta == 0:
            return self.previous_scd
        
        scd_delta = 0
        for i in self.charged_res:
            if sequence[i] in CHARGE:
                charge_at_i = CHARGE[sequence[i]]
                scd_delta += ch_delta * charge_at_i * sqrt(abs(self.mutation_pos - i))

        return self.previous_scd + scd_delta / len(sequence)
    
    def mock_mutation(self, mutation_from: str, mutation_pos: int):
        """
        Update the mutation data, but not the SCD data.
        
        This is called before a `featurizer.featurize` call.
        """
        self.mutation_from = mutation_from
        self.mutation_pos = mutation_pos

    def advance_mutation(self, mutation_pos: int, mutation_to: str, scd: float):
        """
        Update the SCD data.

        This is called before the next round of `mock_mutation`s.
        """
        if mutation_to not in CHARGE:
            self.charged_res.discard(mutation_pos)
        else:
            self.charged_res.add(mutation_pos)
        self.previous_scd = scd

    def clone(self, *, shallow: bool = False):
        """
        Return a copy of self.
        
        The `shallow` parameter (bool) indicates a shallow copy if true
        and otherwise indicates a deepcopy.
        """
        return_value = ScdMachine(self.previous_scd, self.charged_res, self.mutation_from, self.mutation_pos)
        if shallow: # TODO: huh? Isn't already a shallow copy? Also contradicts what happens in 'clone_from'.
            return_value.charged_res = copy(self.charged_res)
        return return_value
    
    def clone_from(self, other, *, shallow: bool = False):
        """
        Write all the data of `other` into self.
        
        The `shallow` parameter (bool) indicates a shallow copy if true
        and otherwise indicates a deepcopy.
        """
        self.previous_scd = other.previous_scd

        if shallow:
            self.charged_res = other.charged_res
        else:
            self.charged_res = copy(other.charged_res)

        self.mutation_from = other.mutation_from
        self.mutation_pos = other.mutation_pos
