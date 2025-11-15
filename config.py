import os

# Absolute path to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Database path
DATABASE_PATH = os.path.join(PROJECT_ROOT, "database", "churn.db")

# Data folder
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
