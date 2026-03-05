from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assuming X_train, X_test, y_train, y_test are already loaded from data_split.py
from data_split import X_train, X_test, y_train, y_test  

print("="*60)
print("COMPREHENSIVE MODEL TESTING - BLOOD PRESSURE ANALYSIS")
print("="*60)

# 1. Logistic Regression
print("\n1. LOGISTIC REGRESSION")
print("-" * 40)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# 2. Decision Tree Classifier
print("\n2. DECISION TREE CLASSIFIER")
print("-" * 40)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# 3. Random Forest Classifier
print("\n3. RANDOM FOREST CLASSIFIER")
print("-" * 40)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# 4. Support Vector Machine (SVM)
print("\n4. SUPPORT VECTOR MACHINE (SVM)")
print("-" * 40)
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

# 5. K-Nearest Neighbors (KNN)
print("\n5. K-NEAREST NEIGHBORS (KNN)")
print("-" * 40)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

# 6. Ridge Classifier
print("\n6. RIDGE CLASSIFIER")
print("-" * 40)
rc = RidgeClassifier()
rc.fit(X_train, y_train)
y_pred_rc = rc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rc))
print("Classification Report:\n", classification_report(y_test, y_pred_rc))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rc))

# 7. Naive Bayes (Selected Model)
print("\n7. NAIVE BAYES (SELECTED MODEL)")
print("-" * 40)
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))

# Summary
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
models = {
    'Logistic Regression': accuracy_score(y_test, y_pred_lr),
    'Decision Tree': accuracy_score(y_test, y_pred_dt),
    'Random Forest': accuracy_score(y_test, y_pred_rf),
    'SVM': accuracy_score(y_test, y_pred_svm),
    'KNN': accuracy_score(y_test, y_pred_knn),
    'Ridge Classifier': accuracy_score(y_test, y_pred_rc),
    'Naive Bayes': accuracy_score(y_test, y_pred_nb),
}

for model_name, accuracy in sorted(models.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name}: {accuracy:.4f}")

best_model = max(models, key=models.get)
print(f"\nBest Model: {best_model} with accuracy {models[best_model]:.4f}")
