import pandas as pd

from sklearn.model_selection import train_test_split


DATA_PATH = "data/raw/data.csv"


# cleaning
def index_date(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df


def clean_raw(df):
    df = index_date(df)
    df = df.dropna()
    return df


# feature engineering (rsi,ema,sma,macd)


def rsi(df, n=14):
    close = df["Close"]
    delta = close.diff()
    delta = delta[1:]

    gain = delta.copy()
    loss = delta.copy()

    gain[gain < 0] = 0
    loss[loss > 0] = 0

    roll_up = gain.rolling(n).mean()
    roll_down = loss.abs().rolling(n).mean()

    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


def ema_sma(df):
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean().shift()
    df["SMA_5"] = df["Close"].rolling(5).mean().shift()
    df["SMA_10"] = df["Close"].rolling(10).mean().shift()
    df["SMA_15"] = df["Close"].rolling(15).mean().shift()
    df["SMA_30"] = df["Close"].rolling(30).mean().shift()

    return df


def macd(df):
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


# clean


def add_features(df):
    """Tambahkan semua fitur teknikal (RSI, EMA, SMA, MACD) untuk inferensi."""
    df = df.copy()
    df = rsi(df)
    df = ema_sma(df)
    df = macd(df)
    return df


def create_label(df):
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


def clean_data(df):
    return df.iloc[33:-1].copy()


# split


def run_split(df, valid_size=0.15):
    drop_cols = ["Date", "Volume", "Open", "Low", "High"]

    df = df.drop(columns=drop_cols)

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=valid_size,
        shuffle=False,
    )

    return X_train, y_train, X_valid, y_valid


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df = clean_raw(df)
    df = rsi(df)
    df = ema_sma(df)
    df = macd(df)
    df = create_label(df)
    df = clean_data(df)

    df.to_csv("data/processed/clean_data.csv")
