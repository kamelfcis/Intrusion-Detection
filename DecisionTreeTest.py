from flask import Flask, request, jsonify
import pandas as pd
import joblib
import webbrowser
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Allow CORS for '/predict' endpoint from all origins

# Load the trained model
model = joblib.load('DecisionTreeClassifier.pkl')
webbrowser.open_new_tab('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data
    data = request.get_json()
    
    # Convert the JSON data to a DataFrame with a default index
    df = pd.DataFrame(data, index=[0])
    
    # Encode categorical variables
    df_encoded = pd.get_dummies(df)  # Assuming one-hot encoding
    
    # Make predictions
    predictions = model.predict(df_encoded)
    
    # Return the predictions as JSON
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
