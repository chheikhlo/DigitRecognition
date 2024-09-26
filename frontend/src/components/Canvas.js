import React, { useRef, useState, useEffect } from 'react';
import axios from 'axios';

const Canvas = () => {
  const canvasRef = useRef(null);
  const [predictedDigit, setPredictedDigit] = useState(null);
  // const [confidence, setConfidence] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctx.strokeStyle = 'white';
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', endDrawing);
    canvas.addEventListener('mouseout', endDrawing);
    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', endDrawing);

    return () => {
      canvas.removeEventListener('mousedown', startDrawing);
      canvas.removeEventListener('mousemove', draw);
      canvas.removeEventListener('mouseup', endDrawing);
      canvas.removeEventListener('mouseout', endDrawing);
      canvas.removeEventListener('touchstart', startDrawing);
      canvas.removeEventListener('touchmove', draw);
      canvas.removeEventListener('touchend', endDrawing);
    };
  }, []);

  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();

    setIsDrawing(true);
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = (e) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const endDrawing = () => {
    setIsDrawing(false);
  };

  const getCanvasPixels = (canvas) => {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);

    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    const grayscale = [];

    for (let i = 0; i < data.length; i += 4) {
      const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      grayscale.push(avg / 255);
    }

    console.log('Grayscale pixels:', grayscale);
    return grayscale;
  };

  const handlePredict = () => {
    const canvas = canvasRef.current;
    const pixels = getCanvasPixels(canvas);
    console.log('Pixels:', pixels);

    axios.post('http://localhost:5000/api/predict_digit', { pixels })
      .then(response => {
        console.log('Response:', response.data);
        setPredictedDigit(response.data.prediction);
        // setConfidence(response.data.confidence);
      })
      .catch(error => {
        console.error('Erreur lors de la prédiction:', error);
      });
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPredictedDigit(null);
    // setConfidence(null);
  };

  return (
    <div>
      <canvas ref={canvasRef} width={280} height={280}
        style={{ border: '1px solid black', marginBottom: '10px' }}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={endDrawing}
        onMouseOut={endDrawing}
        onTouchStart={startDrawing}
        onTouchMove={draw}
        onTouchEnd={endDrawing}
      />
      <div>
        <button onClick={handlePredict}>Prédire</button>
        <button onClick={clearCanvas}>Effacer</button>
      </div>
      {predictedDigit !== null && (
        <div>
          <h2>Prediction:</h2>
          <p style={{ fontWeight: 'bold', fontSize: '60px' }}>{predictedDigit}</p>
          {/* <p>Confiance: {confidence.toFixed(2)}%</p> */}
        </div>
      )}
    </div>
  );
};

export default Canvas;
