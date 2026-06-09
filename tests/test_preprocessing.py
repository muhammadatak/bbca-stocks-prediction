"""Tests untuk modul preprocessing."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Tambahkan src ke path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import (
    clean_data,
    clean_raw,
    create_label,
    ema_sma,
    index_date,
    macd,
    rsi,
    run_split,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Dataframe sample dengan 60 baris data harga."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    close = 9000 + np.cumsum(np.random.randn(60) * 50)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close + np.random.randn(60) * 10,
            "High": close + np.abs(np.random.randn(60) * 20),
            "Low": close - np.abs(np.random.randn(60) * 20),
            "Close": close,
            "Volume": np.random.randint(100000, 500000, 60),
        }
    )


class TestIndexDate:
    def test_index_date_converts_date_column(self, sample_df):
        result = index_date(sample_df.copy())
        assert isinstance(result.index, pd.DatetimeIndex)
        assert "Date" not in result.columns

    def test_index_date_sorted(self, sample_df):
        result = index_date(sample_df.copy())
        assert result.index.is_monotonic_increasing


class TestCleanRaw:
    def test_clean_raw_removes_na(self, sample_df):
        df = sample_df.copy()
        df.loc[df.index[10], "Close"] = np.nan
        result = clean_raw(df)
        assert result["Close"].isna().sum() == 0

    def test_clean_raw_returns_dataframe(self, sample_df):
        result = clean_raw(sample_df.copy())
        assert isinstance(result, pd.DataFrame)


class TestRSI:
    def test_rsi_column_exists(self, sample_df):
        df = index_date(sample_df.copy())
        result = rsi(df)
        assert "RSI" in result.columns

    def test_rsi_values_in_range(self, sample_df):
        df = index_date(sample_df.copy())
        result = rsi(df)
        rsi_values = result["RSI"].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()


class TestEmaSma:
    def test_ema_sma_columns_exist(self, sample_df):
        df = index_date(sample_df.copy())
        result = ema_sma(df)
        for col in ["EMA_9", "SMA_5", "SMA_10", "SMA_15", "SMA_30"]:
            assert col in result.columns


class TestMACD:
    def test_macd_columns_exist(self, sample_df):
        df = index_date(sample_df.copy())
        result = macd(df)
        assert "MACD" in result.columns
        assert "MACD_signal" in result.columns


class TestCreateLabel:
    def test_create_label_binary(self, sample_df):
        df = index_date(sample_df.copy())
        result = create_label(df)
        assert "target" in result.columns
        assert result["target"].dropna().isin([0, 1]).all()

    def test_create_label_has_na_last_row(self, sample_df):
        df = index_date(sample_df.copy())
        result = create_label(df)
        # Baris terakhir seharusnya NaN karena shift(-1)
        assert pd.isna(result["target"].iloc[-1])


class TestCleanData:
    def test_clean_data_no_na(self, sample_df):
        df = index_date(sample_df.copy())
        df = create_label(df)
        result = clean_data(df)
        assert result.isna().sum().sum() == 0


class TestRunSplit:
    def test_run_split_shapes(self, sample_df):
        df = index_date(sample_df.copy())
        df = create_label(df)
        df = clean_data(df)
        X_train, y_train, X_valid, y_valid = run_split(df, valid_size=0.2)
        assert len(X_train) > len(X_valid)
        assert len(y_train) > len(y_valid)
        assert X_train.shape[1] == X_valid.shape[1]

    def test_run_split_no_leakage(self, sample_df):
        df = index_date(sample_df.copy())
        df = create_label(df)
        df = clean_data(df)
        X_train, _, X_valid, _ = run_split(df, valid_size=0.2)
        # Karena shuffle=False, index valid harus > index train
        assert X_valid.index[0] > X_train.index[-1]
