import pandas as pd

def save_csv(df):
    if df is None or len(df) == 0:
        return None
    file_path = "data.csv"
    df.to_csv(file_path, index=False)
    return file_path
