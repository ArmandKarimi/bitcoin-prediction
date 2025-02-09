import os

# Set variables
# Configure BigQuery
PROJECT_ID = "bitcoinclassifier"
FUNCTION_NAME = "update_bitcoin_bigquery"
REGION = "europe-central2"  # Change to your chosen region
SOURCE_DIR = "/Users/armand/Desktop/ML/05_TimeSeries/05_bitcoin/bitcoin-prediction/bitcoin-model-classification/bitcoin-model-classification/src"
 # Path where main.py is stored

# Deploy Cloud Function using a shell command
os.system(f"""
    gcloud functions deploy {FUNCTION_NAME} \
    --runtime python39 \
    --trigger-http \
    --region={REGION} \
    --source={SOURCE_DIR} \
    --allow-unauthenticated
""")
