import yfinance as yf
import pandas as pd
import os
import logging
from datetime import datetime

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/fetch_data_{datetime.now().strftime('%Y%m%d')}.log"

# Set up logging to both file & console
logging.basicConfig(
    level=logging.INFO,  # Log INFO and above (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename),  # Save logs to a file
        logging.StreamHandler()  # Print logs to the console
    ]
)

def fetch_bitcoin_data(start_date="2014-01-01", end_date=None, save_csv=True):
    """
    Fetch historical Bitcoin price data from Yahoo Finance.

    :param start_date: Start date for data retrieval (YYYY-MM-DD)
    :param end_date: End date for data retrieval (YYYY-MM-DD) (default: today's date)
    :param save_csv: Whether to save the data as a CSV file
    :return: Pandas DataFrame containing historical Bitcoin price data
    """
    try:
        logging.info(f"🔄 Fetching Bitcoin data from {start_date} to {end_date or 'today'}")

        # Use today's date as the default end_date if none is provided
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Fetch data from Yahoo Finance
        btc = yf.Ticker("BTC-USD")
        df = btc.history(start=start_date, end=end_date)

        # Keep only relevant columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.reset_index(inplace=True)  # Ensure 'Date' is a column

        if df.empty:
            logging.warning("⚠ No data was fetched! Check the date range.")
            return None

        logging.info(f"✅ Successfully fetched {len(df)} rows of Bitcoin data.")

        # Save to CSV if needed
        if save_csv:
            os.makedirs("../..data/raw", exist_ok=True)  # Ensure directory exists
            file_path = f"../../data/raw/bitcoin_prices_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(file_path, index=False)
            logging.info(f"✅ 📁 Data saved to {file_path}")

        return df

    except Exception as e:
        logging.error(f"❌ Error fetching Bitcoin data: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    # Fetch data from Jan 1, 2022, to today's date
    df = fetch_bitcoin_data(start_date="2014-01-01")
    print(df.info())
