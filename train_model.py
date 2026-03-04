import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Sample dataset for hypertension prediction
# Features: Age, BMI, Heart Rate, Glucose
# Target: 0 = No Hypertension, 1 = Hypertension

# Generate sample data (replace this with your actual dataset)
np.random.seed(42)
n_samples = 1000

age = np.random.randint(20, 80, n_samples)
bmi = np.random.uniform(18, 40, n_samples)
heart_rate = np.random.randint(60, 120, n_samples)
glucose = np.random.uniform(70, 200, n_samples)

# Create target variable based on simple rules (for demonstration)
hypertension = ((age > 45) & (bmi > 30) | (glucose > 140) | (heart_rate > 90)).astype(int)

# Create DataFrame
data = pd.DataFrame({
    'age': age,
    'bmi': bmi,
    'heart_rate': heart_rate,
    'glucose': glucose,
    'hypertension': hypertension
})

# Split features and target
X = data[['age', 'bmi', 'heart_rate', 'glucose']]
y = data['hypertension']

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training Accuracy: {train_score:.2f}")
print(f"Testing Accuracy: {test_score:.2f}")

# Save the model
with open('logreg_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("\nModel saved as 'logreg_model.pkl'")
