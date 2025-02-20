# constants.py
import os
from pathlib import Path

# Define the root directory as the parent of the current file's directory
ROOT_DIR = Path(__file__).resolve().parent 

# Define other paths relative to the root directory
DATA_DIRECTORY = ROOT_DIR / '../../data'
SCORE_DB_DIR = os.path.join(DATA_DIRECTORY, "ScoreDBs")

# sr/lr tag meanings (16 features) for 8 LLPhyScore physical factors.
feature_tagABs = {
    "S2.SUMPI": ["srpipi", "lrpipi"],
    "S3.WATER.V2": ["Water", "Carbon"],
    "S4.SSPRED": ["ssH", "ssE", "ssL"],
    "S5.DISO": ["disL", "disS"],
    "S6.CHARGE.V2": ["srELEC", "lrELEC"],
    "S7.ELECHB.V2": ["sr_hb", "lr_hb"],
    "S8.CationPi.V2": ["srCATPI", "lrCATPI"],
    "S9.LARKS.V2": ["larkSIM", "larkFAR"],
}

feature_tagABs_verbose = {
    "S2.SUMPI": ["pi-pi (short-range)", "pi-pi (long-range)"],
    "S3.WATER.V2": ["protein-water", "protein-carbon"],
    "S4.SSPRED": ["sec. structure (helices)", "sec. structure (strands)"],
    "S5.DISO": ["disorder (long)", "disorder (short)"],
    "S6.CHARGE.V2": ["electrostatic (short-range)", "electrostatic (long-range)"],
    "S7.ELECHB.V2": ["hydrogen bond (short-range)", "hydrogen bond (long-range)"],
    "S8.CationPi.V2": ["cation-pi (short-range)", "cation-pi (long-range)"],
    "S9.LARKS.V2": ["K-Beta similarity", "K-Beta non-similarity"],
}