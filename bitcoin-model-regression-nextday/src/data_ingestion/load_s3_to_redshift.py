import psycopg2
import logging
from datetime import datetime
from dotenv import load_dotenv
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Load environment variables
load_dotenv()

# Redshift Connection Details
IAM_ROLE = os.getenv("IAM_ROLE")  
REDSHIFT_HOST = os.getenv("REDSHIFT_HOST")
REDSHIFT_PORT = os.getenv("REDSHIFT_PORT")
REDSHIFT_DB = os.getenv("REDSHIFT_DB")
REDSHIFT_USER = os.getenv("REDSHIFT_USER")
REDSHIFT_PASSWORD = os.getenv("REDSHIFT_PASSWORD")

TABLE_NAME = "bitcoin_price"
BUCKET_NAME = "my-bitcoin-data-bucket-paris"
S3_KEY = f"data/bitcoin_prices_{datetime.now().strftime('%Y%m%d')}.csv"

def load_data_to_redshift():
    """Loads only new Bitcoin prices from S3 into Redshift, preventing duplicates and updating existing records."""
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

        # Step 1: Create a temporary staging table
        staging_table = f"{TABLE_NAME}_staging"
        create_staging_query = f"""
        CREATE TEMP TABLE {staging_table} (
            id BIGINT IDENTITY(1,1),
            date TIMESTAMP NOT NULL,
            open_price DECIMAL(18,8),
            high_price DECIMAL(18,8),
            low_price DECIMAL(18,8),
            close_price DECIMAL(18,8),
            volume BIGINT
        ) SORTKEY(date);
        """
        cur.execute(create_staging_query)
        logger.info("🆕 Temporary staging table created.")

        # Step 2: Copy new data into the staging table
        copy_query = f"""
        COPY {staging_table}(date, open_price, high_price, low_price, close_price, volume)
        FROM 's3://{BUCKET_NAME}/{S3_KEY}'
        IAM_ROLE '{IAM_ROLE}'
        CSV IGNOREHEADER 1;
        """
        cur.execute(copy_query)
        logger.info("✅ Data copied into the staging table.")

        # Step 3: Merge data - Update if exists, Insert if new
        merge_query = f"""
        BEGIN;

        -- Update existing records
        UPDATE {TABLE_NAME}
        SET open_price = s.open_price,
            high_price = s.high_price,
            low_price = s.low_price,
            close_price = s.close_price,
            volume = s.volume
        FROM {staging_table} s
        WHERE {TABLE_NAME}.date = s.date;

        -- Insert new records
        INSERT INTO {TABLE_NAME} (date, open_price, high_price, low_price, close_price, volume)
        SELECT date, open_price, high_price, low_price, close_price, volume
        FROM {staging_table}
        WHERE date NOT IN (SELECT date FROM {TABLE_NAME});

        COMMIT;
        """
        cur.execute(merge_query)
        logger.info("🔄 Existing records updated, and new records inserted.")

        # Step 4: Drop the staging table
        cur.execute(f"DROP TABLE {staging_table};")
        logger.info("🗑️ Staging table dropped.")

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
