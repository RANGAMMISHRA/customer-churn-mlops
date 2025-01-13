import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_preprocessing import load_and_clean_data

# Load and preprocess data
data = load_and_clean_data("data/churn_data.csv")
X = data.drop("churn", axis=1)
y = data["churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate and log with MLflow
accuracy = accuracy_score(y_test, y_pred)
mlflow.log_metric("accuracy", accuracy)
mlflow.sklearn.log_model(model, "logistic_regression_model")
