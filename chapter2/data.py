import io
import zipfile
import requests
import pandas as pd
import yfinance as yf


def fetch_aapl():
    """
    Fetches Apple stock data from Yahoo Finance and calculates returns.
    """
    # Fetch Apple stock data
    aapl = yf.download("AAPL", start="2020-01-01", end="2023-12-31", progress=False)
    aapl["returns"] = aapl["Close"].pct_change() * 100
    return aapl


def fetch_bike_sharing():
    """
    Downloads and prepares the Bike Sharing dataset from UCI.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # Load the specific file from the zip
    df_bike = pd.read_csv(z.open("hour.csv"))

    # Construct the datetime index
    df_bike["datetime"] = pd.to_datetime(
        df_bike["dteday"] + " " + df_bike["hr"].astype(str) + ":00:00"
    )
    df_bike.set_index("datetime", inplace=True)

    return df_bike
