"""Project-wide paths, constants, and feature-column definitions."""
from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT_DIR / "data" / "raw" / "final_dataset.csv"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

for _d in (MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reproducibility ----------------------------------------------------
RANDOM_STATE = 42

# --- Target --------------------------------------------------------------
# Full-time result recomputed from FTHG/FTAG as a true 3-class label,
# rather than the dataset's own pre-binarized H/NH FTR column.
TARGET_COL = "Result"
TARGET_CLASSES = ["A", "D", "H"]  # alphabetical, matches LabelEncoder default

# --- Columns dropped before modelling ------------------------------------
# FTHG/FTAG/FTR directly encode the outcome (leakage). HTFormPtsStr/
# ATFormPtsStr duplicate the numeric HTFormPts/ATFormPts already present.
# Date is kept only for chronological splitting, not as a model feature.
LEAKAGE_COLS = ["FTHG", "FTAG", "FTR"]
REDUNDANT_COLS = ["HTFormPtsStr", "ATFormPtsStr"]
DATE_COL = "Date"

CATEGORICAL_COLS = [
    "HomeTeam", "AwayTeam",
    "HM1", "HM2", "HM3", "HM4", "HM5",
    "AM1", "AM2", "AM3", "AM4", "AM5",
]

NUMERIC_COLS = [
    "HTGS", "ATGS", "HTGC", "ATGC", "HTP", "ATP", "MW",
    "HTFormPts", "ATFormPts",
    "HTWinStreak3", "HTWinStreak5", "HTLossStreak3", "HTLossStreak5",
    "ATWinStreak3", "ATWinStreak5", "ATLossStreak3", "ATLossStreak5",
    "HTGD", "ATGD", "DiffPts", "DiffFormPts",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# --- Chronological split --------------------------------------------------
# Matches are sorted by Date; the most recent slice is held out as test so
# no future match ever informs a past prediction (same principle as the
# TimeSeriesSplit CV used on the train slice).
TEST_FRACTION = 0.2
CV_SPLITS = 5

# --- Design-system palette (dataviz skill reference palette) -------------
PALETTE = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    "sequential_blue": ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
}

MODEL_NAMES = [
    "LogisticRegression",
    "RandomForest",
    "LightGBM",
    "SVM",
    "MLP (PyTorch)",
]
