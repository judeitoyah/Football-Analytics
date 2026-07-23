"""Loading, target recomputation, and chronological train/test splitting."""
import pandas as pd

from . import config


def load_raw_data() -> pd.DataFrame:
    """Read the raw CSV and parse the (mixed-format) Date column."""
    df = pd.read_csv(config.DATA_RAW, index_col=0)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], format="mixed", dayfirst=True)
    return df


def _true_result(row: pd.Series) -> str:
    """Recompute the real full-time result (H/D/A) from the goal columns.

    The dataset's own FTR column is pre-binarized to H/NH, which throws away
    the draw class. FTHG/FTAG are still present, so the honest 3-class
    target is recovered directly from them.
    """
    if row["FTHG"] > row["FTAG"]:
        return "H"
    if row["FTHG"] < row["FTAG"]:
        return "A"
    return "D"


def build_dataset() -> pd.DataFrame:
    """Load raw data, attach the 3-class target, and drop leakage columns."""
    df = load_raw_data()
    df[config.TARGET_COL] = df.apply(_true_result, axis=1)
    df = df.sort_values(config.DATE_COL).reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame):
    """Split by Date order: earliest rows train, most recent rows test.

    Mirrors the TimeSeriesSplit approach used on other projects — no future
    match is ever allowed to inform a prediction about an earlier one.
    """
    n_test = int(len(df) * config.TEST_FRACTION)
    train_df = df.iloc[: len(df) - n_test].copy()
    test_df = df.iloc[len(df) - n_test :].copy()

    X_train = train_df[config.FEATURE_COLS]
    y_train = train_df[config.TARGET_COL]
    X_test = test_df[config.FEATURE_COLS]
    y_test = test_df[config.TARGET_COL]
    return X_train, X_test, y_train, y_test
