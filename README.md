# In silico benchmarking of featurization strategies for de novo intrinsically disordered protein sequence design

## About

We explore how existing sequence-based featurization schemes for IDPs/IDRs can be extended using the PDB-derived physical-feature-based sequence representations from the LLPhyScore phase separation predictor. We then benchmark the capacity of different featurization schemes to encode functional information by running a suite of subcellular localization predictors on sequences designed using each featurization strategy. 

---

## Installation

Steps to install and set up the project locally:

```bash
# Example
git clone https://github.com/jackiedseal/llfizz.git
cd llfizz
conda create --name llfizz_env python=3.10.13
conda activate llfizz_env
conda install --yes --file requirements.txt
pip install -e . # Ensure llfizz itself is importable.
```

If `benchstuff` isn't already installed, comment out the `benchstuff` line in `requirements.txt` and do this:
```bash
git clone https://github.com/alex-tianhuang/benchstuff.git
pip install -e path/to/benchstuff
```
----

### Featurizing an input sequence
```
python3 python3 scripts/featurize.py sequence.fasta output_features.csv feature_set
```

where 
- `sequence.fasta` is the input sequence
- `output_features.csv` is the CSV file to which the feature values for the sequence will be saved
- `feature_set` is one of the values `original`, `hybrid`, `LLPhyScore` and determines which feature set to use.

As a concrete example,
```bash
# Featurize the COX15 IDR with the original feature set
python3 scripts/featurize.py data/COX15_mimics/sequence_designs/idr_only/cox15_input.fasta output.csv original

# Featurize the COX15 IDR with the original feature set (with custom JSON indicating that SCD shouldn't be included)
python3 scripts/featurize.py data/COX15_mimics/sequence_designs/idr_only/cox15_input.fasta output.csv original --feature-file data/feature_config/no-scd-native-features.json

# Featurize the COX15 IDR with the hybrid feature set
python3 scripts/featurize.py data/COX15_mimics/sequence_designs/idr_only/cox15_input.fasta output.csv hybrid   

# Featurize the COX15 IDR with the LLPhyScore feature set
python3 scripts/featurize.py data/COX15_mimics/sequence_designs/idr_only/cox15_input.fasta output.csv LLPhyScore
```

### Design a sequence using chosen feature set
```bash
python3 scripts/design_feature_mimics.py target_sequence.fasta origin_weights.csv output.csv feature_set --n-random n
```
where
- `target_sequence.fasta` is the target sequence that designs should mimic
- `origin_weights.csv` is a CSV used to weight different elements of the feature vectors based on the mean and variance of that feature in the DisProt proteome
- `output.csv` is the CSV file with the designed sequence and features
- `feature_set` is one of the values `original`, `hybrid`, `LLPhyScore` and determines which feature set to use
- `n-random` is the number of designs to generate (ie. number of random seeds to use)

```bash
# Design two sequences with the original feature set
python3 scripts/design_feature_mimics.py data/COX15_mimics/sequence_designs/idr_only/cox15_input.fasta data/IDRome/disprot_idrome_origin_weights_original_fts.csv output.csv original --n-random 2

# Design one sequence with the hybrid feature set
python3 scripts/design_feature_mimics.py data/COX15_mimics/sequence_designs/idr_only/cox15_input.fasta data/IDRome/disprot_idrome_origin_weights_hybrid_fts.csv output.csv hybrid --n-random 1

# Design one sequence with the LLPhyScore feature set 
# NOTE: This is slower to converge than the other two methods.
python3 scripts/design_feature_mimics.py data/COX15_mimics/sequence_designs/idr_only/cox15_input.fasta data/IDRome/disprot_idrome_origin_weights_llphys_only.csv output.csv LLPhyScore --n-random 1
```


