import yfinance as yf
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.transfers.s3_to_redshift import S3ToRedshiftOperator
from datetime import datetime, timedelta
import logging
import pandas as pd

### ------------- DAG Configuration --------------- ###
S3_BUCKET = "my-bitcoin-data-bucket-paris"
S3_KEY = "raw_data/bitcoin_prices.csv"
REDSHIFT_TABLE = "bitcoin_prices_raw"

# Ensure Redshift connection matches Airflow UI connection ID
AWS_CONN_ID = "aws_default"
REDSHIFT_CONN_ID = "redshift_default"

### ------------- Functions --------------- ###

# Fetch Bitcoin Data from Yahoo Finance
def fetch_bitcoin_data():
    try:
        logging.info("Fetching Bitcoin data...")
        btc = yf.Ticker("BTC-USD")
        # gets only the last value
        df = btc.history(period="1d") 
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.reset_index(inplace = True)
        # Convert 'Date' to string format `YYYY-MM-DD`
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')  # Removes time & timezone
        # Round numeric values to 8 decimal places
        numeric_cols = ['Open', 'High', 'Low', 'Close']
        df[numeric_cols] = df[numeric_cols].round(4)  # Limits decimal places to 8


        if df.empty:
            raise ValueError("No data fetched from Yahoo Finance!")

        df.to_csv('/tmp/bitcoin_data.csv', index=False) 
        logging.info("✅ Bitcoin data saved to /tmp/bitcoin_data.csv")
    except Exception as e:
        logging.error(f"❌ Error fetching Bitcoin data: {str(e)}", exc_info=True)
        raise

# Upload Data to S3
def upload_to_s3():
    try:
        logging.info("Uploading data to S3...")
        s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        s3_hook.load_file('/tmp/bitcoin_data.csv', key=S3_KEY, bucket_name=S3_BUCKET, replace=True)
        logging.info(f"✅ Successfully uploaded data to S3: s3://{S3_BUCKET}/{S3_KEY}")

    except Exception as e:
        logging.error(f"❌ Failed to upload to S3: {str(e)}", exc_info=True)
        raise

### ------------- Airflow DAG Definition --------------- ###
with DAG(
    dag_id="bitcoin_price_daily",
    start_date=datetime(2024, 2, 25),
    schedule_interval='0 23 * * *',  # ✅ Run daily at 11 PM
    catchup=False,
    retries=3,
    retry_delay=timedelta(minutes=5)
) as dag:

    # Step 1: Fetch Bitcoin data and save to CSV
    fetch_data = PythonOperator(
        task_id="fetch_bitcoin_data",
        python_callable=fetch_bitcoin_data
    )

    # Step 2: Upload the CSV file to S3
    upload_data = PythonOperator(
        task_id="upload_to_s3",
        python_callable=upload_to_s3
    )

    # Step 3: Load data from S3 into Redshift
    load_to_redshift = S3ToRedshiftOperator(
        task_id="load_data_to_redshift",
        schema="public",
        table=REDSHIFT_TABLE,
        s3_bucket=S3_BUCKET,
        s3_key=S3_KEY,
        copy_options=['CSV', 'IGNOREHEADER 1'],
        aws_conn_id=AWS_CONN_ID,
        redshift_conn_id=REDSHIFT_CONN_ID
    )


    # add a dbt run
    dbt_run = BashOperator(
    task_id='run_dbt',
    bash_command='cd ~/dbt_bitcoin/bitcoin_prediction && dbt run --debug'
)

    # DAG Task Dependencies
    fetch_data >> upload_data >> load_to_redshift >> dbt_run
