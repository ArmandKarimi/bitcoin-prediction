import boto3
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def upload_to_s3():
    # Specify your AWS S3 bucket name and the file path with current date
    bucket_name = 'my-bitcoin-data-bucket'  # Your actual S3 bucket name
    file_path = f"../../data/raw/bitcoin_prices_{datetime.now().strftime('%Y%m%d')}.csv"  # Path to your local CSV file with current date
    s3_key = f"data/bitcoin_prices_{datetime.now().strftime('%Y%m%d')}.csv"  # S3 key (path) with current date
    print(file_path)
    # Set up the S3 client with your AWS credentials
    s3 = boto3.client('s3')

    try:
        # Upload the file to S3
        s3.upload_file(file_path, bucket_name, s3_key)
        logger.info(f"✅ File uploaded to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logger.error(f"❌ Error uploading file: {e}")

# Call the function to upload to S3
upload_to_s3()
