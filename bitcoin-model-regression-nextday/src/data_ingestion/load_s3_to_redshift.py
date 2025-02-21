import psycopg2
import logging
from datetime import datetime
from dotenv import load_dotenv
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Redshift Connection Details
# Get the root project directory (bitcoin-model-regression-nextday)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Debugging: Print the expected .env path
env_path = os.path.join(BASE_DIR, ".env")

# Load the .env file from the root directory
load_dotenv(env_path)

# Get Redshift credentials
IAM_ROLE = os.getenv("IAM_ROLE")  # For IAM authentication
REDSHIFT_HOST = os.getenv("REDSHIFT_HOST")
REDSHIFT_PORT = os.getenv("REDSHIFT_PORT")
REDSHIFT_DB = os.getenv("REDSHIFT_DB")
REDSHIFT_USER = os.getenv("REDSHIFT_USER")
REDSHIFT_PASSWORD = os.getenv("REDSHIFT_PASSWORD")
IAM_ROLE = os.getenv("IAM_ROLE")

TABLE_NAME = "bitcoin_price"
BUCKET_NAME = "my-bitcoin-data-bucket-paris"
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
