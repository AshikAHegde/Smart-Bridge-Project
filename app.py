from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    bmi = float(request.form["bmi"])
    heart_rate = float(request.form["heart_rate"])
    glucose = float(request.form["glucose"])

    features = np.array([[age, bmi, heart_rate, glucose]])

    # Scale features
    features = scaler.transform(features)

    # Prediction
    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "High Risk of Hypertension"
    else:
        result = "Low Risk of Hypertension"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)