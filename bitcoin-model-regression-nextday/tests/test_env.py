from dotenv import load_dotenv
import os

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

# Print (for testing, remove in production)
print(f"Connected to Redshift at {REDSHIFT_HOST}")
