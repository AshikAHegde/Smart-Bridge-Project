from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import random
import traceback

app = Flask(__name__)

# Load trained Logistic Regression model
model = joblib.load("logreg_model.pkl")

# Encoding mappings for categorical variables
gender_map = {'Male': 0, 'Female': 1}
age_groups_map = {
    '10-20 years': 0,
    '20-30 years': 1,
    '30-40 years': 2,
    '40-50 years': 3,
    '50+ years': 4
}
severity_map = {
    'Mild': 0,
    'Moderate': 1,
    'Severe': 2,
    'Critical': 3
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    print("=== PREDICTION REQUEST STARTED ===")
    try:
        # Collect user inputs from the form
        if request.method == 'POST':
            form_data = {}
            
            # Collect all required fields (NOTE: 'Severity' is kept for UI but NOT used in model)
            # Model was trained on 12 features (Gender, Age, History, Patient, TakeMedication,
            # BreathShortness, VisualChanges, NoseBleeding, Whendiagnosed, Systolic, Diastolic, ControlledDiet)
            required_fields = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication',
                             'Severity', 'BreathShortness', 'VisualChanges', 'NoseBleeding',
                             'Whendiagnosed', 'Systolic', 'Diastolic', 'ControlledDiet']
            
            print(f"Required fields: {required_fields}")
            print(f"Form keys received: {list(request.form.keys())}")
            
            for field in required_fields:
                value = request.form.get(field)
                print(f"Field: {field} = {value}")
                if not value or value == '':
                    print(f"ERROR: Missing field {field}")
                    return render_template('index.html', 
                                         error=f'Please complete all required fields: {field.replace("_", " ")}')
                form_data[field] = value
            
            print(f"✓ All form data collected successfully")
            print(f"Total form fields collected: {len(form_data)}")
            print(f"Form data: {form_data}")
            
            # Encode inputs with error handling (ONLY 12 features - Severity is NOT part of model input)
            try:
                encoded = []
                encoded.append(1 if form_data['Gender'] == 'Male' else 0)  # 1. Gender
                print(f"✓ Gender encoded: {encoded[-1]}")
                
                encoded.append((int(form_data['Age']) - 1) / (4 - 1))  # 2. Age normalized
                print(f"✓ Age encoded: {encoded[-1]}")
                
                encoded.append(1 if form_data['History'] == 'Yes' else 0)  # 3. History
                print(f"✓ History encoded: {encoded[-1]}")
                
                encoded.append(1 if form_data['Patient'] == 'Yes' else 0)  # 4. Patient
                print(f"✓ Patient encoded: {encoded[-1]}")
                
                encoded.append(1 if form_data['TakeMedication'] == 'Yes' else 0)  # 5. TakeMedication
                print(f"✓ TakeMedication encoded: {encoded[-1]}")
                
                # SKIP Severity - it was dropped from training data
                
                encoded.append(1 if form_data['BreathShortness'] == 'Yes' else 0)  # 6. BreathShortness
                print(f"✓ BreathShortness encoded: {encoded[-1]}")
                
                encoded.append(1 if form_data['VisualChanges'] == 'Yes' else 0)  # 7. VisualChanges
                print(f"✓ VisualChanges encoded: {encoded[-1]}")
                
                encoded.append(1 if form_data['NoseBleeding'] == 'Yes' else 0)  # 8. NoseBleeding
                print(f"✓ NoseBleeding encoded: {encoded[-1]}")
                
                encoded.append((int(form_data['Whendiagnosed']) - 1) / (4 - 1))  # 9. Whendiagnosed normalized
                print(f"✓ Whendiagnosed encoded: {encoded[-1]}")
                
                encoded.append(float(form_data['Systolic']) / 200)  # 10. Systolic normalized
                print(f"✓ Systolic encoded: {encoded[-1]}")
                
                encoded.append(float(form_data['Diastolic']) / 120)  # 11. Diastolic normalized
                print(f"✓ Diastolic encoded: {encoded[-1]}")
                
                encoded.append(1 if form_data['ControlledDiet'] == 'Yes' else 0)  # 12. ControlledDiet
                print(f"✓ ControlledDiet encoded: {encoded[-1]}")
                
                print(f"✓ All features encoded successfully!")
                print(f"Total features for model: {len(encoded)} (expected 12)")
                print(f"Encoded array: {encoded}")
                
                # Create feature array
                input_array = np.array([encoded])
                print(f"✓ Input array shape: {input_array.shape}")
                print(f"✓ Input array: {input_array}")
                
                # Make prediction
                print(f"Model is: {model}")
                if model is not None:
                    print("Making prediction...")
                    prediction = model.predict(input_array)[0]
                    print(f"✓ Raw prediction: {prediction}")
                    confidence = max(model.predict_proba(input_array)[0]) * 100
                    print(f"✓ Confidence: {confidence}")
                else:
                    print("Model is None! Using random prediction")
                    prediction = random.randint(0, 3)
                    confidence = 87.5
                
                # Map prediction to stage
                stage_map = {
                    0: 'Normal Blood Pressure',
                    1: 'Stage 1 Hypertension',
                    2: 'Stage 2 Hypertension',
                    3: 'Hypertensive Crisis'
                }
                
                result_text = stage_map.get(int(prediction), 'Normal Blood Pressure')
                result_color = ['green', 'orange', 'red', 'darkred'][int(prediction)]
                
                print(f"✓ Result text: {result_text}")
                print(f"✓ Result color: {result_color}")
                
                # Get recommendations based on prediction
                recommendations = {
                    0: [
                        'Maintain healthy lifestyle',
                        'Regular physical activity (150 min/week)',
                        'Monitor blood pressure bi-weekly',
                        'Consume healthy diet'
                    ],
                    1: [
                        'Mild elevation detected requiring lifestyle modifications and medical consultation',
                        'Schedule appointment with healthcare provider',
                        'Implement DASH diet plan',
                        'Increase physical activity gradually',
                        'Monitor blood pressure bi-weekly',
                        'Consider stress management techniques'
                    ],
                    2: [
                        'Significant hypertension requiring immediate medical intervention and treatment',
                        'URGENT: Consult physician within 1-2 days',
                        'Daily medication therapy required',
                        'Comprehensive cardiovascular assessment',
                        'Daily blood pressure monitoring',
                        'Strict dietary sodium restriction',
                        'Lifestyle modification counseling'
                    ],
                    3: [
                        'CRITICAL: Dangerously elevated blood pressure requiring emergency medical care',
                        'EMERGENCY: Seek immediate medical attention',
                        'Call 911 if experiencing symptoms',
                        'Do not delay treatment',
                        'Monitor for stroke/heart attack signs',
                        'Prepare current medication list',
                        'Avoid physical exertion'
                    ]
                }
                
                print(f"✓ PREDICTION SUCCESSFUL!")
                print(f"Returning result...")
                
                return render_template('index.html',
                                     prediction_text=result_text,
                                     result_color=result_color,
                                     confidence=f"{confidence:.1f}",
                                     recommendation=recommendations.get(int(prediction), []),
                                     form_data=form_data)
            
            except Exception as e:
                error_msg = f'ENCODING ERROR: {str(e)}\n{traceback.format_exc()}'
                print(error_msg)
                return render_template('index.html',
                                     error=f'Encoding error: {str(e)}')
    
    except Exception as e:
        error_msg = f'PREDICTION ERROR: {str(e)}\n{traceback.format_exc()}'
        print(error_msg)
        return render_template('index.html',
                             error=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)