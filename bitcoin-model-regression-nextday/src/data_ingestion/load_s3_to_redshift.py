import psycopg2
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Redshift Connection Details
IAM_ROLE = "arn:aws:iam::842676012982:role/service-role/AmazonRedshift-CommandsAccessRole-20250208T175217"
REDSHIFT_HOST = "bitcoin-cluster.cppa05dpkktd.eu-north-1.redshift.amazonaws.com"
REDSHIFT_PORT = "5439"
REDSHIFT_DB = "dev"
REDSHIFT_USER = "admin"
REDSHIFT_PASSWORD = "Carryme5702!"
TABLE_NAME = "bitcoin_prices"
BUCKET_NAME = "my-bitcoin-data-bucket"
S3_KEY = f"data/bitcoin_prices_{datetime.now().strftime('%Y%m%d')}.csv"

def load_data_to_redshift():
    """Loads the latest Bitcoin prices from S3 into Redshift."""
    conn = None
    try:
        # Connect to Redshift
        conn = psycopg2.connect(
            dbname=REDSHIFT_DB,
            user=REDSHIFT_USER,
            password=REDSHIFT_PASSWORD,
            host=REDSHIFT_HOST,
            port=REDSHIFT_PORT
        )
        cur = conn.cursor()

        # Copy new data from S3 into Redshift
        copy_query = f"""
        COPY {TABLE_NAME}
        FROM 's3://{BUCKET_NAME}/{S3_KEY}'
        IAM_ROLE '{IAM_ROLE}'
        CSV IGNOREHEADER 1;
        """
        cur.execute(copy_query)
        logger.info("✅ Data copied from S3 to Redshift.")

        conn.commit()
        cur.close()
        logger.info("🚀 Data successfully updated in Redshift!")

    except Exception as e:
        logger.error(f"❌ Error loading data into Redshift: {e}")
    finally:
        if conn:
            conn.close()

# Run the function
load_data_to_redshift()
