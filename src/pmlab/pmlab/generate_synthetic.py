import json
import joblib
import numpy as np
import pandas as pd

from . import config, db

def generate_synthetic_stream(eng, n_points=2000, seed=42) -> pd.DataFrame:
    config.ensure_dirs()
    train_df = db.fetch_training(eng)
    resid_df = pd.read_csv(config.OUTPUTS / "train_residuals.csv")

    with open(config.OUTPUTS / "thresholds.json", "r", encoding="utf-8") as f:
        thr = json.load(f)

    dt = config.median_dt_seconds(train_df[config.DB_TIME])
    T = float(thr["T_seconds"])
    pts_T = max(1, int(round(T / dt)))

    t0 = float(train_df[config.DB_TIME].min())
    times = t0 + np.arange(n_points) * dt

    rng = np.random.default_rng(seed)
    out = pd.DataFrame({config.DB_TIME: times})

    for axis in config.DB_AXIS_COLS:
        model = joblib.load(config.MODELS / f"{axis}_lr.joblib")
        pred = model.predict(times.reshape(-1, 1))

        axis_resid = resid_df[resid_df["axis"] == axis]["residual"].dropna().values
        if len(axis_resid) < 50:
            noise = rng.normal(0, 1, size=n_points)
        else:
            noise = rng.choice(axis_resid, size=n_points, replace=True)

        y = pred + noise

        # Inject 2–4 anomaly segments
        minc = float(thr["MinC"][axis])
        maxc = float(thr["MaxC"][axis])
        segs = int(rng.integers(2, 5))

        for _ in range(segs):
            length = int(rng.integers(pts_T, pts_T * 3 + 1))
            start = int(rng.integers(0, max(1, n_points - length)))
            end = start + length

            offset = (minc * rng.uniform(1.1, 1.6)) if rng.random() < 0.5 else (maxc * rng.uniform(1.1, 1.6))
            y[start:end] = y[start:end] + offset

        out[axis] = y

    out.to_csv(config.OUTPUTS / "synthetic_stream.csv", index=False)
    return out
