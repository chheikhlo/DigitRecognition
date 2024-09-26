from flask import Flask, request, jsonify
import numpy as np
from joblib import load
import os
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'best_digit_recognition_model.pkl')

def load_model():
    model = load(model_path)
    return model

client = MongoClient('mongodb+srv://ndeye:ndeyeIpssi@atlascluster.mp58qo5.mongodb.net/?retryWrites=true&w=majority&appName=AtlasCluster')
db = client['digit_recognition']
collection = db['predict']

@app.route('/api/predict_digit', methods=['POST'])
def predict_digit():
    try:
        data = request.get_json()
        pixels = data['pixels']
        print('Pixels reçus:', pixels[:100])

        if len(pixels) != 784:
            return jsonify({'error': 'Invalid input data'}), 400

        input_data = np.array(pixels).reshape(1, -1)
        print('Input data:', input_data)

        model = load_model()
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        confidence = np.max(prediction_proba) * 100
        print('Prédiction:', prediction)
        print('Confiance:', confidence)

        # Sauvegarde dans MongoDB
        prediction_data = {
            'pixels': pixels,
            'prediction': int(prediction[0]),
            'confidence': confidence
        }
        collection.insert_one(prediction_data)

        return jsonify({'prediction': int(prediction[0]), 'confidence': confidence})

    except Exception as e:
        print(f"Erreur lors de la prédiction: {str(e)}")
        return jsonify({'error': 'Erreur lors de la prédiction'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
