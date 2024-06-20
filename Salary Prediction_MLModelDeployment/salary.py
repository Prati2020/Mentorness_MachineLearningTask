import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load your pre-trained models (replace with your actual model loading logic)
models = {
    'Random Forest': RandomForestRegressor(max_depth=10, min_samples_leaf=4, n_estimators=10),
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_leaf=2, min_samples_split=10),
    'GradientBoost': GradientBoostingRegressor(min_samples_split=10, n_estimators=50)
}

# Helper function for data preprocessing (replace with your actual preprocessing steps)
def preprocess_data(data):
    # Handle missing values, scaling, categorical encoding, etc.
    # Replace with your specific preprocessing logic based on your data
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    categorical_features = []  # Identify categorical features if any
    preprocessor = ColumnTransformer([
        ('impute', imputer, []),
        ('scale', scaler, []),
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # For categorical features
    ])
    return preprocessor.fit_transform(data)

# Create the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Load data from JSON (replace with appropriate logic for your data format)
        X = pd.DataFrame(data)

        # Preprocess data
        X_preprocessed = preprocess_data(X)

        # Make predictions using the chosen model
        model_name = request.args.get('model', default='Random Forest')
        if model_name not in models:
            return jsonify({'error': f'Invalid model name: {model_name}'}), 400
        model = models[model_name]
        predictions = model.predict(X_preprocessed)

        # Return predictions as JSON
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
