from pathlib import Path
import pandas as pd
import numpy as np

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root
DATA_RAW = ROOT / "data" / "raw" / "RMBR4-2_export_test.csv"

OUTPUTS = ROOT / "outputs"
PLOTS = OUTPUTS / "plots"
MODELS = ROOT / "models"

TRAINING_TABLE = "training_data"
STREAMING_TABLE = "streaming_data"
EVENTS_TABLE = "anomaly_events"

# RUBRIC: Axes #1–#8
AXES_USED = list(range(1, 9))

CSV_TIME = "Time"
CSV_AXIS = lambda i: f"Axis #{i}"

DB_TIME = "time"
DB_AXIS = lambda i: f"axis{i}"

DB_AXIS_COLS = [DB_AXIS(i) for i in AXES_USED]

def ensure_dirs():
    OUTPUTS.mkdir(exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)

def parse_time_to_elapsed_seconds(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    if dt.isna().any():
        raise ValueError("Some Time values could not be parsed. Check timestamp format.")
    t0 = dt.iloc[0]
    return (dt - t0).dt.total_seconds().astype(float)

def load_training_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    needed = [CSV_TIME] + [CSV_AXIS(i) for i in AXES_USED]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Keep only needed cols, rename to DB format
    rename_map = {CSV_TIME: DB_TIME}
    rename_map.update({CSV_AXIS(i): DB_AXIS(i) for i in AXES_USED})

    df = df[needed].rename(columns=rename_map)

    # Convert timestamp -> elapsed seconds
    df[DB_TIME] = parse_time_to_elapsed_seconds(df[DB_TIME])

    # Coerce axes to numeric
    for c in DB_AXIS_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=DB_AXIS_COLS, how="all").sort_values(DB_TIME).reset_index(drop=True)
    return df

def median_dt_seconds(time_series: pd.Series) -> float:
    vals = time_series.astype(float).values
    diffs = np.diff(vals)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return 1.0
    return max(float(np.median(diffs)), 1e-6)
