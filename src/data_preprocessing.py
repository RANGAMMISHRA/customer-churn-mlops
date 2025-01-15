import pandas as pd

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    return df
