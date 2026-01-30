from src.data_loader import load_data
from src.preprocessing import preprocess_data

df = load_data()
X, y = preprocess_data(df)

print("Feature matrix shape:", X.shape)
print("Target distribution:")
print(y.value_counts())
