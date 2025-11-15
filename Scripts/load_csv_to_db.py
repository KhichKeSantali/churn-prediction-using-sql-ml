import os
import pandas as pd
from sqlalchemy import create_engine
from config import DATABASE_PATH

# Load CSV data into a SQLite database
data_path = os.path.join(os.getcwd(), '..', 'data', 'Telco-Customer-Churn.csv')
df = pd.read_csv(data_path)

# Create a SQLite database connection
try:
    db_path = os.path.join(os.getcwd(), 'database', 'churn.db')
    engine = create_engine(f"sqlite:///{DATABASE_PATH}")
    print(f"Data loaded into the database successfully. {len(df)} records inserted.")
except Exception as e:
    print(f"Error creating database connection: {e}")
    exit(1)


# Load data into the 'customers' table
df.to_sql('customers', engine, index=False, if_exists='replace')

