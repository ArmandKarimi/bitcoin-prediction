import yfinance as yf
from datetime import datetime, timedelta

def fetch_bitcoin_data(start_date=None, end_date=None):
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    btc = yf.Ticker("BTC-USD")
    df = btc.history(start=start_date, end=end_date)

    print(df)  # Check if data is fetched correctly

fetch_bitcoin_data()
