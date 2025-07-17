from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained models and supporting files
try:
    print("Loading models and support files...")
    rf_model = joblib.load('random_forest_model.joblib')
    nb_model = joblib.load('naive_bayes_model.joblib')
    feature_names = joblib.load('feature_names.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    print("Models and support files loaded successfully")
except Exception as e:
    print(f"Error loading models and support files: {str(e)}")
    raise

# Load symptom severity data
severity_df = pd.read_csv('Symptom-severity.csv')
severity_weights = dict(zip(severity_df['Symptom'], severity_df['weight']))

def predict_disease(symptoms, model):
    try:
        # Create input data with the same columns as training data
        input_data = pd.DataFrame(0, index=[0], columns=feature_names)
        
        print(f"\nPredicting with symptoms: {symptoms}")
        
        # Set the provided symptoms to their severity weights
        for symptom in symptoms:
            if symptom in input_data.columns:
                weight = severity_weights.get(symptom, 1)
                input_data.loc[0, symptom] = weight
                print(f"Setting {symptom} with weight {weight}")
        
        # Make prediction
        prediction_proba = model.predict_proba(input_data)
        
        # Get top 3 predictions with probabilities
        top_3_indices = np.argsort(prediction_proba[0])[-3:][::-1]
        
        top_3_predictions = []
        for idx in top_3_indices:
            disease = label_encoder.inverse_transform([idx])[0]
            probability = float(prediction_proba[0][idx])
            top_3_predictions.append({
                'disease': disease,
                'probability': probability
            })
            print(f"Predicted {disease} with probability {probability:.4f}")
        
        return top_3_predictions
            
    except Exception as e:
        print(f"Error in predict_disease: {str(e)}")
        raise

@app.route('/')
def home():
    # Get list of symptoms from feature names
    symptoms = [symptom for symptom in feature_names]
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_symptoms = request.form.getlist('symptoms')
        print(f"\nReceived symptoms: {selected_symptoms}")
        
        if not selected_symptoms:
            return jsonify({'error': 'Please select at least one symptom'})
        
        # Get predictions from both models
        print("\nGetting Random Forest predictions...")
        rf_predictions = predict_disease(selected_symptoms, rf_model)
        
        print("\nGetting Naive Bayes predictions...")
        nb_predictions = predict_disease(selected_symptoms, nb_model)
        
        response_data = {
            'random_forest': rf_predictions,
            'naive_bayes': nb_predictions,
            'selected_symptoms': selected_symptoms
        }
        
        print("\nSending response:", response_data)
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error in predict route: {error_msg}")
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
