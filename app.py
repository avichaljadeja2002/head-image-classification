# app.py
import os
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from classifier import create_combined_model, orientation_map
from tensorflow.keras.models import Model, load_model

# Initialize Flask app
app = Flask(__name__)

# Load the model
combined_model = load_model("combined_model.h5")
# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    file_path = data.get("file_path")
    print("here to predict file " + file_path )
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        img = Image.open(file_path).convert('RGB')
        img = img.resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Get predictions
        print("going to predict")
        decoded_img, predictions = combined_model.predict(img_array)
        print(predictions)
        predicted_label = np.argmax(predictions[0])
        orientation = orientation_map[predicted_label]

        return jsonify({
            "file_path": file_path,
            "predicted_orientation": orientation,
            "prediction_confidence": float(predictions[0][predicted_label])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
