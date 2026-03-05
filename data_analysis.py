import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_excel('./Dataset/patient_data.xlsx')
data.head()

# Check for null values
print("Null values:")
print(data.isnull().sum())

# Rename columns
data.rename(columns={'C': 'Gender', 'Whendiagnoused': 'Whendiagnosed'}, inplace=True)

# Strip accidental leading/trailing spaces in categorical values
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].str.strip()

# Clean and standardize data values
data['TakeMedication'] = data['TakeMedication'].replace({'Yes ': 'Yes'})
data['NoseBleeding'] = data['NoseBleeding'].replace({'No ': 'No'})
data['Systolic'] = data['Systolic'].replace({'121-130': '121 - 130'})
data['Systolic'] = data['Systolic'].replace({'100-120': '100 - 110', '100-120 ': '100 - 110'})
data['Stages'] = data['Stages'].replace({'HYPERTENSION (Stage 2)': 'HYPERTENSION (Stage-2)'})
data['Stages'] = data['Stages'].replace({'HYPERTENSIVE CRISIS ': 'HYPERTENSIVE CRISIS'})

# Check for specific values in Diastolic column
print(f"Count of '130' in Diastolic: {(data['Diastolic'] == '130').sum()}")
print(f"Count of '100' in Diastolic: {(data['Diastolic'] == '100').sum()}")

# Fix Diastolic values
data['Diastolic'] = data['Diastolic'].replace({'130': '100'})

# Check for duplicates
print(f"Number of duplicate rows: {data.duplicated().sum()}")

# Remove duplicates
data.drop_duplicates(inplace=True)

# Display cleaned data
print("\nCleaned data:")
print(data)

# ============================================
# LABEL ENCODING
# ============================================

# Define nominal and ordinal features
nominal_features = ['Gender', 'History', 'Patient', 'TakeMedication', 'BreathShortness', 'VisualChanges', 'NoseBleeding', 'ControlledDiet', 'Severity']
ordinal_features = [f for f in data.columns if f not in nominal_features]
ordinal_features.remove('Stages')
print(f"Nominal features: {nominal_features}")
print(f"Ordinal features: {ordinal_features}")

# Encode nominal features
print("\n--- Label Encoding Applied ---")

# Binary features: No=0, Yes=1
for col in nominal_features:
    if set(data[col].unique()) == {'Yes', 'No'}:
        data[col] = data[col].map({'No': 0, 'Yes': 1})
    elif col == 'Gender':
        data[col] = data[col].map({'Male': 0, 'Female': 1})

# Age groups: 18-34=1, 35-50=2, 51-64=3, 65+=4
data['Age'] = data['Age'].map({'18-34': 1, '35-50': 2, '51-64': 3, '65+': 4})

# Severity: Mild=0, Moderate=1, Severe=2
data['Severity'] = data['Severity'].replace({'Mild': 0, 'Moderate': 1, 'Severe': 2})

# When Diagnosed: <1 Year=1, 1-5 Years=2, >5 Years=3
data['Whendiagnosed'] = data['Whendiagnosed'].map({'<1 Year': 1, '1 - 5 Years': 2, '>5 Years': 3})

# Blood pressure ranges - Systolic: 100-110=0, 111-120=1, 121-130=2, 130+=3
data['Systolic'] = data['Systolic'].map({'100 - 110': 0, '111 - 120': 1, '121 - 130': 2, '130+': 3})

# Blood pressure ranges - Diastolic: 70-80=0, 81-90=1, 91-100=2, 100+=3
data['Diastolic'] = data['Diastolic'].map({'70 - 80': 0, '81 - 90': 1, '91 - 100': 2, '100+': 3})

# Target stages: Normal=0, Stage-1=1, Stage-2=2, Crisis=3
data['Stages'] = data['Stages'].map({'NORMAL': 0, 'HYPERTENSION (Stage-1)': 1, 'HYPERTENSION (Stage-2)': 2, 'HYPERTENSIVE CRISIS': 3})

print("Label encoding completed!")
print(f"\nEncoded data:\n{data.head()}")

# ============================================
# SCALING ORDINAL FEATURES
# ============================================

from sklearn.preprocessing import MinMaxScaler

# Apply MinMaxScaler to ordinal features for optimal model performance
scaler = MinMaxScaler()
data[ordinal_features] = scaler.fit_transform(data[ordinal_features])

print(f"\nScaled data (ordinal features):\n{data.head()}")
print("\n✓ Data preprocessing completed successfully!")
data.to_csv('./Dataset/patient_data_cleaned.csv', index=False)
# or
# data.to_excel('./Dataset/patient_data_cleaned.xlsx', index=False)

