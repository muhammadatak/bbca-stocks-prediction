"""
Data Drift Detector — PSI (Population Stability Index)
------------------------------------------------------
Scenario B: Membandingkan distribusi fitur data baru terhadap baseline.
PSI < 0.1  → no drift
PSI >= 0.1 → drift (trigger retrain)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from preprocess import clean_raw, rsi, ema_sma, macd, create_label, clean_data

PUSHGATEWAY = "localhost:9091"
BASELINE_PATH = ROOT_DIR / "data" / "processed" / "clean_data.csv"
NEW_DATA_PATH = ROOT_DIR / "data" / "raw" / "data.csv"
BINS = 10

FEATURES = [
    "Close", "RSI", "EMA_9", "SMA_5", "SMA_10",
    "SMA_15", "SMA_30", "MACD", "MACD_signal",
]


def _psi_single(expected: np.ndarray, actual: np.ndarray, bins: int = BINS) -> float:
    all_vals = np.concatenate([expected, actual])
    _, edges = np.histogram(all_vals, bins=bins)
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    e_pct = e_hist / len(expected) + 1e-10
    a_pct = a_hist / len(actual) + 1e-10
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def compute_psi(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    common = [f for f in FEATURES if f in baseline_df.columns and f in new_df.columns]
    return {f: _psi_single(baseline_df[f].dropna().values, new_df[f].dropna().values) for f in common}


def main():
    baseline = pd.read_csv(BASELINE_PATH, index_col=0, parse_dates=True)
    new_raw = pd.read_csv(NEW_DATA_PATH)
    new_raw = clean_raw(new_raw)
    new_raw = rsi(new_raw)
    new_raw = ema_sma(new_raw)
    new_raw = macd(new_raw)
    new_raw = create_label(new_raw)
    new_raw = clean_data(new_raw)

    psi = compute_psi(baseline, new_raw)
    avg_psi = float(np.mean(list(psi.values())))
    max_psi = float(np.max(list(psi.values())))

    print("PSI per feature:")
    for k, v in psi.items():
        print(f"  {k:12s} : {v:.4f}")
    print(f"\navg_psi = {avg_psi:.4f}")
    print(f"max_psi = {max_psi:.4f}")

    # push ke pushgateway
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
        registry = CollectorRegistry()
        Gauge("drift_psi_avg", "Average PSI", registry=registry).set(avg_psi)
        Gauge("drift_psi_max", "Max PSI", registry=registry).set(max_psi)
        push_to_gateway(PUSHGATEWAY, job="drift-detector", registry=registry)
        print(f"✅ PSI pushed to {PUSHGATEWAY}")
    except Exception:
        print("⚠️  Pushgateway not available — metrics printed only")

    DRIFT_THRESHOLD = 0.1
    if avg_psi >= DRIFT_THRESHOLD:
        print(f"\n🚨 DRIFT DETECTED (avg_psi={avg_psi:.4f} >= {DRIFT_THRESHOLD})")
        sys.exit(1)
    else:
        print(f"\n✅ No drift (avg_psi={avg_psi:.4f} < {DRIFT_THRESHOLD})")
        sys.exit(0)


if __name__ == "__main__":
    main()
