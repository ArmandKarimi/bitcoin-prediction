import psycopg2
import os
from dotenv import load_dotenv

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

# Define the SQL query to create the table
create_table_query = """
CREATE TABLE bitcoin_price (
    id BIGINT IDENTITY(1,1),
    date TIMESTAMP NOT NULL,
    open_price DECIMAL(18,8),
    high_price DECIMAL(18,8),
    low_price DECIMAL(18,8),
    close_price DECIMAL(18,8),
    volume BIGINT,
    PRIMARY KEY (id)
)
SORTKEY(date);
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
    cur.execute(create_table_query)
    conn.commit()
    
    print("✅ Table 'bitcoin_price' created successfully!")
    
    cur.close()
    conn.close()
except Exception as e:
    print(f"❌ Error creating table: {e}")
