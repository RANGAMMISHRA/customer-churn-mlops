import pytest
from src.data_preprocessing import load_and_clean_data
from src.train_model import LogisticRegression, train_test_split

def test_model_training():
    # Load sample data
    data = load_and_clean_data("data/churn_data.csv")
    X = data.drop("churn", axis=1)
    y = data["churn"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Ensure the model can make predictions
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)

    # Check the model accuracy is a valid float between 0 and 1
    accuracy = model.score(X_test, y_test)
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0

def test_model_initialization():
    # Ensure model initializes correctly
    model = LogisticRegression()
    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")

def test_data_loading():
    # Ensure data loading works
    data = load_and_clean_data("data/churn_data.csv")
    assert not data.isnull().values.any()
    assert "churn" in data.columns
