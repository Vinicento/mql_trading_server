from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pickled model (ensure your model is pickled as .pkl file)
model = joblib.load('decision_tree_model.pkl')  # Path to your pickled model


# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request (expects JSON with features)
    data = request.get_json(force=True)

    # Extract features from the JSON. For example, assuming we need 'stochastic_12' and 'kama_148'
    try:
        stochastic_12 = float(data['stochastic_12'])
        kama_148 = float(data['kama_148'])
    except (KeyError, ValueError):
        return jsonify({'error': 'Invalid input. Provide both stochastic_12 and kama_148 as float values.'}), 400

    # Create a DataFrame from the input data
    input_data = pd.DataFrame([[stochastic_12, kama_148]], columns=['stochastic_12', 'kama_148'])

    # Make the prediction using the loaded model
    prediction = model.predict(input_data)[0]

    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction)})


# Run the Flask app
if __name__ == '__main__':

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
