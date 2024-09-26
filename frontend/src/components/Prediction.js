import React from 'react';
const Prediction = ({ predictedDigit }) => {
    return (
      <div>
        <h2>Résultat de la prédiction:</h2>
        <p>Chiffre prédit: {predictedDigit}</p>
      </div>
    );
  };
export default Prediction;
