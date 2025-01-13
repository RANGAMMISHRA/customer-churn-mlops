from src.data_preprocessing import load_and_clean_data

def test_load_and_clean_data():
    df = load_and_clean_data("data/churn_data.csv")
    assert not df.isnull().values.any()
    assert "churn" in df.columns
