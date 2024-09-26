import React, { useState } from 'react';
import Canvas from './components/Canvas';
import Prediction from './components/Prediction';
import "./App.css";

function App() {
  const [prediction, setPrediction] = useState(null);

  // Fonction pour mettre à jour la prédiction
  const handlePrediction = (predictedValue) => {
    setPrediction(predictedValue);
  };

  return (
    <div className="App">
      <h1>Reconnaissance de chiffres avec le modèle RandomForest</h1>
      <Canvas onPredict={handlePrediction} />
      {prediction !== null && <Prediction prediction={prediction} />}
    </div>
  );
}

export default App;
