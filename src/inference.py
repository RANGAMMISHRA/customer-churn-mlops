import mlflow
import pandas as pd
import joblib
from mlflow.tracking import MlflowClient

# Dynamically fetch the latest model and scaler
def load_model_and_scaler():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Customer-Churn-Experiment")
    latest_run = client.search_runs(experiment.experiment_id, order_by=["start_time desc"], max_results=1)[0]
    latest_run_id = latest_run.info.run_id

    # Load the model using the latest run ID
    model_uri = f"runs:/{latest_run_id}/logistic_regression_model_with_signature"
    model = mlflow.sklearn.load_model(model_uri)

    # Load the scaler separately from the artifact location
    scaler_path = f"./scaler.pkl"
    scaler = joblib.load(scaler_path)

    return model, scaler

# Function to make predictions using the latest model
def make_predictions(input_data):
    model, scaler = load_model_and_scaler()
    
    # Preprocess input data using the loaded scaler
    input_scaled = scaler.transform(input_data)

    # Make predictions
    predictions = model.predict(input_scaled)
    return predictions

# Test the inference script
if __name__ == "__main__":
    test_data = pd.DataFrame({
        "CustomerID": [1001, 1002],
        "Age": [34, 42],
        "Tenure": [12, 24],
        "Usage Frequency": [15, 22],
        "Support Calls": [1, 0],
        "Payment Delay": [0, 1],
        "Total Spend": [1200, 2500],
        "Last Interaction": [20, 5],
        "Gender_Male": [1, 0],
        "Subscription Type_Premium": [0, 1],
        "Subscription Type_Standard": [1, 0],
        "Contract Length_Monthly": [1, 0],
        "Contract Length_Quarterly": [0, 1]
    })

    predictions = make_predictions(test_data)
    print(f"\nPredicted Churn: {predictions}")
