"""
Simulasi — generate shifted data untuk testing Scenario A & B.
Menambahkan noise/guncangan pada fitur technical indicator
untuk mensimulasikan perubahan distribusi data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed" / "clean_data.csv"
OUTPUT = ROOT / "data" / "raw" / "data.csv"


def generate_shifted_data(noise_level: float = 0.3):
    """Tambahkan noise ke fitur agar distribusi berubah (simulasi drift)."""
    df = pd.read_csv(INPUT, index_col=0)

    feature_cols = ["Close", "RSI", "EMA_9", "SMA_5", "SMA_10",
                    "SMA_15", "SMA_30", "MACD", "MACD_signal"]

    shifted = df.copy()
    for col in feature_cols:
        if col in shifted.columns:
            std = shifted[col].std()
            noise = np.random.normal(0, noise_level * std, size=len(shifted))
            shifted[col] = shifted[col] + noise

    # Re-calculate target after shifting
    shifted["target"] = (shifted["Close"].shift(-1) > shifted["Close"]).astype(int)
    shifted = shifted.iloc[:-1]

    shifted.to_csv(OUTPUT)
    print(f"✅ Shifted data saved to {OUTPUT}")
    print(f"   Noise level: {noise_level}")
    print(f"   Rows: {len(shifted)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=0.3,
                        help="Noise level (std multiplier)")
    args = parser.parse_args()
    generate_shifted_data(args.noise)
