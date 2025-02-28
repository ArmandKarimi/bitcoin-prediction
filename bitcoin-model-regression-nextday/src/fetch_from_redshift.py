import psycopg2
import logging
import os
from dotenv import load_dotenv
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

load_dotenv()

def load_data():

    # Get Redshift credentials
    REDSHIFT_HOST = os.getenv("REDSHIFT_HOST")
    REDSHIFT_PORT = os.getenv("REDSHIFT_PORT")
    REDSHIFT_DB = os.getenv("REDSHIFT_DB")
    REDSHIFT_USER = os.getenv("REDSHIFT_USER")
    REDSHIFT_PASSWORD = os.getenv("REDSHIFT_PASSWORD")

    # Ensure credentials are available
    if not all([REDSHIFT_HOST, REDSHIFT_PORT, REDSHIFT_DB, REDSHIFT_USER, REDSHIFT_PASSWORD]):
        logging.error("❌ Redshift credentials are missing. Check your environment variables.")
        return None

    # Define the SQL query to fetch data
    test_query = """
    SELECT 
        close_price AS "Close",
        open_price AS "Open",
        high_price AS "High",
        low_price AS "Low",
        volume AS "Volume",
        close_pct AS "Close_pct",
        high_low AS "High_Low",
        open_close AS "Open_Close",
        open_high AS "Open_High",
        open_low AS "Open_Low",
        close_1d AS "Close_1D",
        close_2d AS "Close_2D",
        close_3d AS "Close_3D",
        close_4d AS "Close_4D",
        close_5d AS "Close_5D",
        close_6d AS "Close_6D",
        volume_1d AS "Volume_1D",
        volume_2d AS "Volume_2D",
        ma_3d AS "MA_3D",
        ma_7d AS "MA_7D",
        ma_30d AS "MA_30D",
        day_sin AS "Day sin",
        day_cos AS "Day cos",
        year_sin AS "Year sin",
        year_cos AS "Year cos"
    FROM bitcoin_price_transform;
    """

    # Connect to Redshift and execute the query
    try:
        conn = psycopg2.connect(
            host=REDSHIFT_HOST,
            port=REDSHIFT_PORT,
            dbname=REDSHIFT_DB,
            user=REDSHIFT_USER,
            password=REDSHIFT_PASSWORD
        )
        cur = conn.cursor()
        cur.execute(test_query)
        
        # Fetch the results from the query
        results = cur.fetchall()

        # Fetch column names from cursor description
        columns = [desc[0] for desc in cur.description]

        # Convert results to a DataFrame
        df = pd.DataFrame(results, columns=columns)

        # Define the new column names in the correct format
        new_column_names = [
            "Close", "Open", "High", "Low", "Volume", "Close_pct", "High_Low",
            "Open_Close", "Open_High", "Open_Low", "Close_1D", "Close_2D",
            "Close_3D", "Close_4D", "Close_5D", "Close_6D", "Volume_1D",
            "Volume_2D", "MA_3D", "MA_7D", "MA_30D", "Day sin", "Day cos",
            "Year sin", "Year cos"
        ]

        # Rename the DataFrame columns
        df.columns = new_column_names

        df = df.astype("float32")

        logging.info("✅ Data fetched from Redshift!")  # Corrected logging call

        # Close the cursor and connection
        cur.close()
        conn.close()

        return df
        
    except Exception as e:
        logging.error(f"❌ Error: {e}")  # Use logging.error to capture errors
        return None


# Example usage
if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(df.describe())  # Print sample data
