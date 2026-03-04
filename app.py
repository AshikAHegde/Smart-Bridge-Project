from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('logreg_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        heart_rate = float(request.form['heart_rate'])
        glucose = float(request.form['glucose'])
        
        # Create feature array
        features = np.array([[age, bmi, heart_rate, glucose]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Interpret result
        if prediction[0] == 1:
            result = "High risk of Hypertension"
        else:
            result = "Low risk of Hypertension"
        
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
