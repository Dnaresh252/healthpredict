# Disease Prediction Web Application

A web-based system that predicts diseases based on user-selected symptoms using Machine Learning models (Random Forest and Naive Bayes).

## Features

- Interactive symptom selection with search functionality
- Severity-weighted predictions
- Top 3 disease predictions from each model
- Probability scores for each prediction
- Mobile-responsive design
- Real-time search filtering of symptoms

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have the following files in your project directory:
- `random_forest_model.joblib` (trained Random Forest model)
- `naive_bayes_model.joblib` (trained Naive Bayes model)
- `Symptom-severity.csv` (symptom severity data)

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Use the search box to filter symptoms
2. Select relevant symptoms by checking the checkboxes
3. Click "Predict Disease" to get predictions
4. View the top 3 predictions from both models with confidence scores
5. Selected symptoms will be displayed below the predictions

## Project Structure

```
├── app.py                      # Flask application
├── requirements.txt            # Project dependencies
├── static/
│   └── style.css              # CSS styles
├── templates/
│   └── index.html             # HTML template
├── random_forest_model.joblib  # Trained Random Forest model
├── naive_bayes_model.joblib   # Trained Naive Bayes model
└── Symptom-severity.csv       # Symptom severity data
```

## Technical Details

- The application uses Flask for the backend
- Predictions are weighted based on symptom severity
- Both models return probability scores for predictions
- The frontend is built with vanilla JavaScript and CSS Grid
- Responsive design works on mobile devices

## Note

This is a demonstration system and should not be used as a substitute for professional medical advice. Always consult with healthcare professionals for medical decisions. 