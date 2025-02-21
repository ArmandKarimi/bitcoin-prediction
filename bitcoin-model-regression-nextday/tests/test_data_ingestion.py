import psycopg2
import logging
import os
from dotenv import load_dotenv

# Set up logging to both file & console
logging.basicConfig(
    level=logging.INFO,  # Log INFO and above (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # Print logs to the console
        logging.FileHandler("app.log")  # Save logs to a file called app.log
    ]
)

# Redshift connection details
# Get the root project directory (bitcoin-prediction)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

# Load the .env file from the root directory
load_dotenv(os.path.join(BASE_DIR, ".env"))


# Get Redshift credentials
REDSHIFT_HOST = os.getenv("REDSHIFT_HOST")
REDSHIFT_PORT = os.getenv("REDSHIFT_PORT")
REDSHIFT_DB = os.getenv("REDSHIFT_DB")
REDSHIFT_USER = os.getenv("REDSHIFT_USER")
REDSHIFT_PASSWORD = os.getenv("REDSHIFT_PASSWORD")

# Define the SQL query to fetch data
test_query = """
SELECT * 
FROM bitcoin_price 
LIMIT 10;
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
    
    # Print the results
    for row in results:
        print(row)
    
    logging.info("✅ Table 'bitcoin_price' exists and data fetched!")  # Corrected logging call
    
    # Close the cursor and connection
    cur.close()
    conn.close()
except Exception as e:
    logging.error(f"❌ Error: {e}")  # Use logging.error to capture errors
