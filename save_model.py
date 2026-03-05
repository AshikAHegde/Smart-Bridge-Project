import joblib
from data_split import X_train, X_test, y_train, y_test
from sklearn.linear_model import LogisticRegression

# Train Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# save model
joblib.dump(logreg, "logreg_model.pkl")
print("✅ Model saved as logreg_model.pkl")
