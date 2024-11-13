# app.py
from flask import Flask, jsonify, request
from PIL import Image
import numpy as np
import os
from flask_cors import CORS
from tensorflow.keras.models import load_model
from classifier import head_orientation_map
from person_classifier import people_orientation_map
from sunglasses_classifier import sunglasses_orientation_map
from emotion_classifier import emotion_orientation_map

app = Flask(__name__)
CORS(app)

# Load the models
emotion_combined_model = load_model("emotion_model.h5")
person_classifier_model = load_model("person_combined_model.h5")
glasses_classifier = load_model("sunglasses_model.h5")
orientation_classifier = load_model("head_orientation_combined_model.h5")

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

        decoded_img, orientation_predictions = orientation_classifier.predict(img_array)
        orientation_predicted_label = np.argmax(orientation_predictions[0])
        orientation = head_orientation_map[orientation_predicted_label]
        orientation_confidence = float(orientation_predictions[0][orientation_predicted_label])

        decoded_img, emotion_prediction = emotion_combined_model.predict(img_array)
        emotion_predicted_label = np.argmax(emotion_prediction[0])
        emotion = emotion_orientation_map[emotion_predicted_label]
        emotion_confidence = float(emotion_prediction[0][emotion_predicted_label])

        decoded_img, person_prediction = person_classifier_model.predict(img_array)
        people_predicted_label = np.argmax(person_prediction[0])
        person = people_orientation_map[people_predicted_label]
        person_confidence = float(person_prediction[0][people_predicted_label])

        decoded_img, sunglasses_prediction = glasses_classifier.predict(img_array)
        sunglasses_predicted_label = np.argmax(sunglasses_prediction[0])
        sunglasses = sunglasses_orientation_map[sunglasses_predicted_label]
        sunglasses_confidence = float(sunglasses_prediction[0][sunglasses_predicted_label])

        # Return all predictions in JSON format
        return jsonify({
            "emotion": {
                "predicted_label": emotion,
                "prediction_confidence": emotion_confidence
            },
            "person": {
                "predicted_label": person,
                "prediction_confidence": person_confidence
            },
            "glasses": {
                "predicted_label": sunglasses,
                "prediction_confidence": sunglasses_confidence
            },
            "orientation": {
                "predicted_orientation": orientation,
                "prediction_confidence": orientation_confidence
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
