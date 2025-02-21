import psycopg2
import logging


# Set up logging to both file & console
logging.basicConfig(
    level=logging.INFO,  # Log INFO and above (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler()  # Print logs to the console
    ]
)

conn = psycopg2.connect(
    host="bitcoin-cluster-paris.cjnvc1uvrkvj.eu-west-3.redshift.amazonaws.com",
    port="5439",
    dbname="dev",
    user="admin",
    password="Carryme5702!"
)
logging.info("✅ Connection successful!")
conn.close()
