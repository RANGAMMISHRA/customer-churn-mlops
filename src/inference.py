import mlflow
import pandas as pd

def predict_new_data(model_path, data):
    model = mlflow.sklearn.load_model(model_path)
    predictions = model.predict(data)
    return predictions

new_data = pd.DataFrame({"feature1": [0.5], "feature2": [1.2]})
predictions = predict_new_data("mlflow_experiments/customer_churn_model", new_data)
print("Predictions:", predictions)
