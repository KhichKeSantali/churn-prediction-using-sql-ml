import pandas as pd
from sqlalchemy import create_engine

# Load CSV data into a SQLite database
df = pd.read_csv('data/Telco-Customer-Churn.csv')

# Create a SQLite database connection
try:
    engine = create_engine('sqlite:///database/churn.db')
    print(f"Data loaded into the database successfully. {len(df)} records inserted.")
except Exception as e:
    print(f"Error creating database connection: {e}")
    exit(1)


# Load data into the 'customers' table
df.to_sql('customers', engine, index=False, if_exists='replace')

# Confirm successful load