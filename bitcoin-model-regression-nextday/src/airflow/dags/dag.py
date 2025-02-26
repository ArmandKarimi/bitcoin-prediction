from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.transfers.s3_to_redshift import S3ToRedshiftOperator
from airflow.providers.amazon.aws.operators.sagemaker_training import SageMakerTrainingOperator
from airflow.providers.slack.operators.slack_api import SlackAPIPostOperator
from cosmos.providers.dbt.task_group import DbtTaskGroup
from datetime import datetime
import requests
import json
import pandas as pd
import io

### ---------- Functions ------------ ###

# Check Network Connectivity
def check_network():
    try:
        logging.info("Checking network connectivity...")
        response = requests.get("https://www.google.com")
        if response.status_code == 200:
            logging.info("Network is working fine.")
        else:
            logging.error(f"Network check failed with status code {response.status_code}.")
    except Exception as e:
        logging.error(f"Network check failed: {str(e)}")


# Fetch Bitcoin Data
def fetch_bitcoin_data(start_date="2014-01-01", end_date=None):

    if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        logging.info(f"Starting to fetch Bitcoin data from {start_date} to {end_date}")
        
        # Fetch Bitcoin data from Yahoo Finance
        btc = yf.Ticker("BTC-USD")

        # Get historical data
        logging.info("Fetching data from Yahoo Finance...")
        df = btc.history(start=start_date, end=end_date)

        logging.info(f"Fetched data successfully with {len(df)} rows.")
        
        # Save the data to /tmp folder
        df.to_csv('/tmp/bitcoin_data.csv', index=False)
        logging.info("Data saved to /tmp/bitcoin_data.csv")

        # Return the data in a dictionary format for Airflow
        return df.to_dict(orient='records')

    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)


# Upload Data to S3
def upload_to_s3():
    try:
        logging.info("Uploading Bitcoin data to S3...")
        s3_hook = S3Hook(aws_conn_id='aws_default')
        s3_hook.load_file('/tmp/bitcoin_data.csv', S3_KEY, bucket_name=S3_BUCKET, replace=True)
        logging.info(f"✅ Successfully uploaded Bitcoin data to S3: {S3_BUCKET}/{S3_KEY}")
    except Exception as e:
        logging.error(f"❌ Failed to upload to S3: {str(e)}", exc_info=True)
        raise


### ------------- DAGS --------------- ########

# DAG Configuration
S3_BUCKET = "my-bitcoin-data-bucket-paris"
S3_KEY = "raw_data/bitcoin_prices.csv"
REDSHIFT_TABLE = "bitcoin_prices_raw"
DBT_PROJECT_DIR = "/home/ubuntu/dbt_bitcoin/bitcoin_prediction/"
DBT_PROFILES_DIR = "/home/ubuntu/dbt_bitcoin/bitcoin_prediction"

# Define Airflow DAG
with DAG(
    dag_id="bitcoin_price_prediction",
    start_date=datetime(2024, 2, 25),
    schedule_interval= None, #one-time run
    catchup=False,
) as dag:

    # Step 1: Fetch Bitcoin price data from Yahoo Finance and upload to S3
    fetch_data = PythonOperator(
        task_id="fetch_bitcoin_data",
        python_callable=fetch_bitcoin_data
    )

    # Step 2: Load data from S3 to Redshift
    load_to_redshift = S3ToRedshiftOperator(
        task_id="load_data_to_redshift",
        schema="public",
        table=REDSHIFT_TABLE,
        s3_bucket=S3_BUCKET,
        s3_key=S3_KEY,
        copy_options=['CSV', 'IGNOREHEADER 1'],
        aws_conn_id="aws_default",
        redshift_conn_id="redshift_default"
    )

    # Step 3: Run dbt transformations in Redshift
    dbt_transform = DbtTaskGroup(
        group_id="dbt_transformations",
        project_dir=DBT_PROJECT_DIR,
        profiles_dir=DBT_PROFILES_DIR,
        profile="default",
        target="dev"
    )

    # Step 4: Train SageMaker model on transformed data
    train_sagemaker = SageMakerTrainingOperator(
        task_id="train_sagemaker_model",
        config={
            "TrainingJobName": "bitcoin-price-prediction",
            "AlgorithmSpecification": {
                "TrainingImage": "xxxxxxxxxx.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:latest",
                "TrainingInputMode": "File",
            },
            "RoleArn": "arn:aws:iam::xxxxxx:role/service-role/AmazonSageMaker-ExecutionRole",
            "InputDataConfig": [{"ChannelName": "train", "DataSource": {"S3DataSource": {"S3Uri": f"s3://{S3_BUCKET}/transformed_data/", "S3DataType": "S3Prefix"}}}],
            "OutputDataConfig": {"S3OutputPath": f"s3://{S3_BUCKET}/model_output/"},
            "ResourceConfig": {"InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 10},
            "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
        },
        aws_conn_id="aws_default"
    )

    # Step 5: Send notification via Slack
    notify_slack = SlackAPIPostOperator(
        task_id="send_slack_notification",
        token="your-slack-token",
        channel="#ml-alerts",
        text="Bitcoin price prediction model training completed successfully!",
    )

    # DAG Dependencies
    fetch_data >> load_to_redshift >> dbt_transform >> train_sagemaker >> notify_slack
