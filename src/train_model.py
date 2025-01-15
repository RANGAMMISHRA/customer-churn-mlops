import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from data_preprocessing import load_and_clean_data

# Enable MLflow Experiment
mlflow.set_experiment("Customer-Churn-Experiment")

# Start an MLflow run
with mlflow.start_run():

    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_and_clean_data("data/churn_data.csv")

    # Verify columns to avoid KeyError
    print(f"Columns available: {data.columns}")
    if "Churn" not in data.columns:
        raise ValueError("The column 'Churn' was not found in the dataset.")

    # Split features and target variable
    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    # Apply feature scaling
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and test sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize and train the Logistic Regression model with increased iterations
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=500, solver='lbfgs')
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)

    # ✅ Corrected: Logging the model with input example and signature
    mlflow.sklearn.log_model(
        model, 
        "logistic_regression_model_with_signature",
        signature=mlflow.models.infer_signature(X_train, model.predict(X_train)),
        input_example=X_train[:5]
    )

    # ✅ Fixed: Logging the Scaler Separately as an Artifact Instead of a Model
    import joblib
    joblib.dump(scaler, "scaler.pkl")
    mlflow.log_artifact("scaler.pkl")

print("Model training and logging completed successfully!")
