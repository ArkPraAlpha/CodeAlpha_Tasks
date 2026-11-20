"""Sentiment over time, anomaly detection and root-cause hints."""
import numpy as np
import pandas as pd


def sentiment_over_time(df, score_col="ensemble_score", label_col="sentiment", freq="M", min_n=5):
    d = df.dropna(subset=["date"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["period"] = d["date"].dt.to_period(freq).dt.to_timestamp()
    g = d.groupby("period").agg(
        n=("text", "size"),
        mean_score=(score_col, "mean"),
        pct_positive=(label_col, lambda s: (s == "positive").mean() * 100),
        pct_negative=(label_col, lambda s: (s == "negative").mean() * 100),
    ).reset_index()
    g = g[g["n"] >= min_n].reset_index(drop=True)
    g["rolling_mean"] = g["mean_score"].rolling(3, min_periods=1).mean()
    return g


def detect_anomalies(ts, z_threshold=2.0):
    """Flag periods whose mean sentiment is unusually high/low (z-score)."""
    ts = ts.copy()
    std = ts["mean_score"].std(ddof=0)
    ts["z_score"] = 0.0 if len(ts) < 4 or std == 0 else (ts["mean_score"] - ts["mean_score"].mean()) / std
    ts["anomaly"] = ts["z_score"].abs() >= z_threshold
    ts["direction"] = np.where(ts["z_score"] < 0, "dip", "spike")
    return ts


def explain_anomalies(ts, aspect_df, freq="M", min_mentions=3):
    """For each anomalous period, find the aspect that drove the change."""
    out = []
    for _, row in ts[ts["anomaly"]].iterrows():
        info = {"period": row["period"], "direction": row["direction"],
                "mean_score": row["mean_score"], "z_score": row["z_score"], "driver": None,
                "driver_score": None}
        if not aspect_df.empty:
            a = aspect_df.dropna(subset=["date"])
            in_period = a[a["date"].dt.to_period(freq).dt.to_timestamp() == row["period"]]
            stats = in_period.groupby("aspect")["score"].agg(["mean", "size"])
            stats = stats[stats["size"] >= min_mentions]
            if not stats.empty:
                aspect = stats["mean"].idxmin() if row["direction"] == "dip" else stats["mean"].idxmax()
                info["driver"], info["driver_score"] = aspect, float(stats.loc[aspect, "mean"])
        out.append(info)
    return out
