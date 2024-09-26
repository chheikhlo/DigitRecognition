from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import rpy2.robjects as robjects
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

# Fonction de Chargement du modèle
def load_model():
    robjects.r('library(randomForest)')
    robjects.r('load_model <- function(x_path) { readRDS(x_path) }')
    model = robjects.r['load_model']("best_digit_recognition_model.rds")
    return model

model = load_model()

client = MongoClient('mongodb+srv://ndeye:ndeyeIpssi@atlascluster.mp58qo5.mongodb.net/?retryWrites=true&w=majority&appName=AtlasCluster')
db = client['digit_recognition']
collection = db['predict']

@app.route('/api/predict_digit', methods=['POST'])
def predict_digit():
    # données d'entrée recupérer depuis la requête
    data = request.get_json()
    pixels = data['pixels']

    # Préparer les données pour le modèle
    input_data = np.array(pixels).reshape(1, -1)

    # Creons un DataFrame R avec les noms de colonnes appropriés comme les colonnes dans train.csv
    column_names = [f'pixel{i}' for i in range(784)]  # Les colonnes doivent correspondre à celles du modèle

    # Créer un dictionnaire des données d'entrée
    input_dict = {column_names[i]: robjects.FloatVector(input_data[0, i:i+1]) for i in range(784)}

    # Convertir le dictionnaire en DataFrame R
    input_df = robjects.DataFrame(input_dict)

    # Faire des prédictions
    prediction = robjects.r['predict'](model, newdata=input_df)
    confidence = robjects.r['predict'](model, newdata=input_df, type='prob')

    # Calculer la confiance
    confidence_value = np.max(np.array(confidence)) * 100

    print("Prediction:", prediction[0])
    print("Confidence:", confidence)

    # Sauvegarde dans Atlas
    prediction_data = {
        'pixels': pixels,
        'prediction': int(prediction[0]),
        'confidence': confidence_value
    }
    collection.insert_one(prediction_data)

    return jsonify({'prediction': int(prediction[0]), 'confidence': confidence_value})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
