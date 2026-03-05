import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ------------------------------------------------
# 1 Load Dataset
# ------------------------------------------------
data = pd.read_excel("Dataset/patient_data_cleaned.xlsx")

print("Dataset Shape:", data.shape)
print("Columns:", data.columns)

# ------------------------------------------------
# 2 Features and Target
# ------------------------------------------------
X = data.drop("Stages", axis=1)
y = data["Stages"]

# ------------------------------------------------
# 3 Data Splitting
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------
# 4 Feature Scaling
# ------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------
# 5 Algorithms
# ------------------------------------------------
models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(n_estimators=200, random_state=42),

    "SVM":
        SVC(kernel="rbf"),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "Ridge Classifier":
        RidgeClassifier(),

    "Gaussian Naive Bayes":
        GaussianNB()
}

results = {}

print("\nMODEL COMPARISON\n")

for name, model in models.items():

    if name in ["Logistic Regression", "SVM", "KNN", "Ridge Classifier"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    results[name] = acc

    print("------------")
    print(name)
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

# ------------------------------------------------
# 6 Best Model Selection
# ------------------------------------------------
best_model_name = max(results, key=results.get)

print("\nBest Model:", best_model_name)

best_model = models[best_model_name]

# ------------------------------------------------
# 7 Save Model
# ------------------------------------------------
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved")