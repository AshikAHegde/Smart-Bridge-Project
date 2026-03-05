import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('Dataset/patient_data_cleaned.csv')

# Display data info
print(data.head())
print(data.shape)
print(data.info())

# Drop rows with missing values
data = data.dropna()

# Drop 'Severity' column (string) and 'Stages' (target)
x = data.drop(['Stages', 'Severity'], axis=1)
y = data['Stages']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
