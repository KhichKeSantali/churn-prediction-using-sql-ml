import os
from config import DATABASE_PATH
from config import PROJECT_ROOT
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load Cleaned Data from Database
db_path = os.path.join(os.getcwd(),'..', 'data', 'cleaned_customers.csv')
df = pd.read_csv(db_path)

# Split Data into Features and Target
x = df.drop('Churn', axis=1)
y = df['Churn']

# Split Data into Training and Testing Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale Features (Except for tree models)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Helper function to evaluate models
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return {"Model": model_name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}

# Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(x_train_scaled, y_train)
y_pred_log = log_model.predict(x_test_scaled)
log_results = evaluate_model(y_test, y_pred_log, "Logistic Regression")

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(x_train_scaled, y_train)
y_pred_rf = rf_model.predict(x_test_scaled)
rf_results = evaluate_model(y_test, y_pred_rf, "Random Forest Classifier")

# Compile Results
results_df = pd.DataFrame([log_results, rf_results])
print(results_df)

# Visualize Confusion Matrix for Random Forest
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Random Forest Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Save results to database
engine = create_engine(f'sqlite:///{DATABASE_PATH}')
results_df.to_sql('model_performance', con=engine, if_exists='replace', index=False)
print("Model performance results saved to database.")

# Save the best model

# Create models directory if it doesn't exist
project = PROJECT_ROOT
models_dir = os.path.join(project, 'models')

os.makedirs(models_dir, exist_ok=True)

# Paths for saving models
model_path = os.path.join(models_dir, 'churn_model.pkl')
preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
columns_path = os.path.join(models_dir, 'feature_columns.pkl')

# Save Model
if rf_results['F1 Score'] >= log_results['F1 Score']:
    model = rf_model
else:
    model = log_model
joblib.dump(model, model_path)

# Save preprocessor and model
joblib.dump(scaler, preprocessor_path)

# Save feature columns
joblib.dump(list(x_train.columns), columns_path)

print(f"Best model saved to {model_path}")









