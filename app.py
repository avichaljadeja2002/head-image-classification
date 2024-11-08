# app.py
from flask import Flask, jsonify, request
from PIL import Image
import numpy as np
import os
from flask_cors import CORS
from tensorflow.keras.models import load_model
from classifier import orientation_map

app = Flask(__name__)
CORS(app)
# Load the model
combined_model = load_model("combined_model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Open and process the image
        img = Image.open(file).convert('RGB')
        img = img.resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Get predictions
        decoded_img, predictions = combined_model.predict(img_array)
        predicted_label = np.argmax(predictions[0])
        orientation = orientation_map[predicted_label]

        return jsonify({
            "predicted_orientation": orientation,
            "prediction_confidence": float(predictions[0][predicted_label])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
