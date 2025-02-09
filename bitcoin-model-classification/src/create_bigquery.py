from google.cloud import bigquery

# Configure your Google Cloud project
PROJECT_ID = "bitcoinclassifier"
DATASET_ID = "bitcoin_data"
TABLE_ID = "bitcoin_prices"

# Initialize BigQuery client
client = bigquery.Client(project=PROJECT_ID)

# Create dataset if it doesn't exist
dataset_ref = client.dataset(DATASET_ID)
dataset = bigquery.Dataset(dataset_ref)
dataset.location = "EU"  # Change if needed

try:
    client.create_dataset(dataset, exists_ok=True)
    print(f"✅ Dataset {DATASET_ID} is ready!")
except Exception as e:
    print(f"⚠ Error creating dataset: {e}")

# Create table schema
schema = [
    bigquery.SchemaField("Date", "TIMESTAMP"),
    bigquery.SchemaField("Open", "FLOAT"),
    bigquery.SchemaField("High", "FLOAT"),
    bigquery.SchemaField("Low", "FLOAT"),
    bigquery.SchemaField("Close", "FLOAT"),
    bigquery.SchemaField("Volume", "FLOAT"),
]

# Create table if it doesn't exist
table_ref = client.dataset(DATASET_ID).table(TABLE_ID)
table = bigquery.Table(table_ref, schema=schema)

try:
    client.create_table(table, exists_ok=True)
    print(f"✅ Table {TABLE_ID} is ready in dataset {DATASET_ID}!")
except Exception as e:
    print(f"⚠ Error creating table: {e}")
