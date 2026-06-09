import yfinance as yf
import os
from datetime import datetime, timedelta
import pandas as pd
from preprocess import index_date


DATA_PATH = "data/raw/data.csv"
START_DATE = "2020-01-01"
TODAY_DATE = datetime.today()


def get_last_date():
    df = pd.read_csv(DATA_PATH)
    df = index_date(df)
    df = df.sort_index()
    last_date = df.index[-1]
    return last_date.strftime("%Y-%m-%d")


def ingest_data():

    end_date = (TODAY_DATE + timedelta(days=1)).strftime("%Y-%m-%d")

    # tidak ada data
    if not os.path.exists(DATA_PATH):
        print(
            f"data not found ingest new data from {START_DATE} TO {
                TODAY_DATE.strftime('%Y-%m-%d')
            }"
        )
        df = yf.download(
            tickers="BBCA.JK", start=START_DATE, end=end_date, multi_level_index=False
        )

        df.to_csv(DATA_PATH)
        return

    # ingest data dari data terakhir
    start = get_last_date()
    df_old = pd.read_csv(DATA_PATH)
    df_old = index_date(df_old)

    df_new = yf.download(
        tickers="BBCA.JK", start=start, end=end_date, multi_level_index=False
    )
    if df_new.empty:
        print("data up to date")
        return

    df = pd.concat([df_old, df_new]).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    print(f"ingest data {start} to {TODAY_DATE.strftime('%Y-%m-%d')}")
    df.to_csv(DATA_PATH)


if __name__ == "__main__":
    ingest_data()
