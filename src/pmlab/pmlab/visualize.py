import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

def plot_events(stream_df: pd.DataFrame, events_df: pd.DataFrame):
    config.ensure_dirs()
    times = stream_df[config.DB_TIME].astype(float).values
    X = times.reshape(-1, 1)

    for axis in config.DB_AXIS_COLS:
        model = joblib.load(config.MODELS / f"{axis}_lr.joblib")
        pred = model.predict(X)
        obs = stream_df[axis].astype(float).values

        fig, ax = plt.subplots()
        ax.scatter(times, obs, s=8)
        ax.plot(times, pred)

        if not events_df.empty:
            subset = events_df[events_df["axis"] == axis]
            for _, ev in subset.iterrows():
                ax.axvspan(ev["start_time"], ev["end_time"], alpha=0.2)
                mid = (ev["start_time"] + ev["end_time"]) / 2
                ax.text(mid, np.nanmax(obs), f"{ev['event_type']} ({ev['duration_s']:.1f}s)", rotation=90, va="top")

        ax.set_title(f"{axis}: Regression + Alerts/Errors")
        ax.set_xlabel("time (s, elapsed)")
        ax.set_ylabel(axis)

        fig.tight_layout()
        fig.savefig(config.PLOTS / f"{axis}_events.png", dpi=200)
        plt.close(fig)
