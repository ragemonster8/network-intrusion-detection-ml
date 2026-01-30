from src.data_loader import load_data
from src.preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from src.model import train_logistic_regression

df = load_data()
X, y = preprocess_data(df)

print("Feature matrix shape:", X.shape)
print("Target distribution:")
print(y.value_counts())



# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)



# Train the model
model = train_logistic_regression(X_train, y_train)

print("Model training completed.")
