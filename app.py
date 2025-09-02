from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('breast_cancer_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the feature names in the correct order
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Define the main route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the form
        input_features = [float(request.form[name]) for name in feature_names]

        # Create a numpy array and reshape for a single prediction
        features_array = np.array(input_features).reshape(1, -1)

        # Scale the features
        scaled_features = scaler.transform(features_array)

        # Make a prediction
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)

        # Interpret the prediction
        if prediction[0] == 1:
            result = "Malignant"
            confidence = f"{probability[0][1]*100:.2f}%"
        else:
            result = "Benign"
            confidence = f"{probability[0][0]*100:.2f}%"

        return jsonify({'prediction': result, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
