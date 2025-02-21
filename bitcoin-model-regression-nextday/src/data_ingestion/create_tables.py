import psycopg2

# Redshift connection details
REDSHIFT_HOST = "bitcoin-cluster-paris.cjnvc1uvrkvj.eu-west-3.redshift.amazonaws.com"
REDSHIFT_PORT = "5439"
REDSHIFT_DB = "dev"
REDSHIFT_USER = "admin"
REDSHIFT_PASSWORD = "Carryme5702!"

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
    print("✅ Table 'bitcoin_prices' created successfully!")
    cur.close()
    conn.close()
except Exception as e:
    print(f"❌ Error creating table: {e}")
