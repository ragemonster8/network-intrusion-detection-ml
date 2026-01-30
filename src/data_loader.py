import pandas as pd
import os

def load_data():
    data_path = os.path.join("data", "raw", "nsl_kdd.csv")
    df = pd.read_csv(data_path)
    return df
