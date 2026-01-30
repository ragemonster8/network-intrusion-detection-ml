import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    df = df.copy()

    #Converting label to binary
    df['label'] = df['label'].apply(
        lambda x: 0 if x == 'normal' else 1
    )

    #Identifying categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = categorical_cols.drop('label', errors='ignore')

    #Encoding categorical features
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    #Separating features and target
    X = df.drop('label', axis=1)
    y = df['label']

    #Scaling numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
