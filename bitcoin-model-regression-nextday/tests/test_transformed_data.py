import psycopg2
import logging

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
REDSHIFT_HOST = "bitcoin-cluster.cppa05dpkktd.eu-north-1.redshift.amazonaws.com"
REDSHIFT_PORT = "5439"
REDSHIFT_DB = "dev"  # Database name
REDSHIFT_USER = "admin"  # my username
REDSHIFT_PASSWORD = "Carryme5702!"  # my password

# Define the SQL query to fetch data
test_query = """
SELECT * 
FROM public.bitcoin_transformation
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
    
    logging.info("✅ Table 'bitcoin_prices' exists and data fetched!")  # Corrected logging call
    
    # Close the cursor and connection
    cur.close()
    conn.close()
except Exception as e:
    logging.error(f"❌ Error: {e}")  # Use logging.error to capture errors
