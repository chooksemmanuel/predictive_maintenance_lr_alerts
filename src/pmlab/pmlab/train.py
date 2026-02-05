import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import config, db

def train_models_from_db(eng):
    config.ensure_dirs()
    df = db.fetch_training(eng)

    thresholds = {"T_seconds": None, "MinC": {}, "MaxC": {}}
    dt = config.median_dt_seconds(df[config.DB_TIME])
    thresholds["T_seconds"] = float(5 * dt)

    residual_rows = []

    for axis in config.DB_AXIS_COLS:
        X = df[[config.DB_TIME]].values
        y = df[axis].values
        mask = ~np.isnan(y)
        X2, y2 = X[mask], y[mask]

        model = LinearRegression()
        model.fit(X2, y2)

        # Save model
        joblib.dump(model, config.MODELS / f"{axis}_lr.joblib")

        # Residuals on all rows
        pred = model.predict(X)
        resid = y - pred

        # Store residuals
        for t, r in zip(df[config.DB_TIME].astype(float).values, resid):
            if pd.isna(r):
                continue
            residual_rows.append({"axis": axis, "time": float(t), "residual": float(r)})

        # Thresholds from positive residuals
        r = pd.Series(resid).dropna()
        pos = r[r > 0]
        if len(pos) < 20:
            base = r.abs()
            minc = float(np.percentile(base, 95))
            maxc = float(np.percentile(base, 99))
        else:
            minc = float(np.percentile(pos, 95))
            maxc = float(np.percentile(pos, 99))

        if maxc <= minc:
            maxc = minc * 1.25 if minc > 0 else minc + 1e-6

        thresholds["MinC"][axis] = minc
        thresholds["MaxC"][axis] = maxc

    residual_df = pd.DataFrame(residual_rows)
    residual_df.to_csv(config.OUTPUTS / "train_residuals.csv", index=False)

    with open(config.OUTPUTS / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    return thresholds
