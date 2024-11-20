# app.py
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
import numpy as np
import os
from flask_cors import CORS
from tensorflow.keras.models import load_model
import ssl

app = Flask(__name__)

# Load the models
emotion_combined_model = load_model("emotion_model.h5")
person_classifier_model = load_model("person_combined_model.h5")
glasses_classifier = load_model("sunglasses_model.h5")
orientation_classifier = load_model("head_orientation_combined_model.h5")

head_orientation_map = {0: 'up', 1: 'straight', 2: 'left', 3: 'right'}
sunglasses_orientation_map = {0: 'no_sunglasses', 1: 'sunglasses'}
people_orientation_map = {0: 'an2i', 1: 'at33', 2: 'boland', 3: 'bpm', 4: 'ch4f', 5: 'cheyer', 6: 'choon', 7: 'danieln', 8: 'glickman',9: 'karyadi',10: 'kawamura', 11: 'kk49',12: 'megak',13: 'mitchell', 14: 'night', 15: 'phoebe',16: 'saavik',17: 'steffi',18: 'sz24',19: 'tammo'}
emotion_orientation_map = {0: 'angry', 1: 'happy', 2: 'sad', 3: 'neutral'}

@app.route('/')
def serve_html():
    return send_from_directory('.', 'website.html')

@app.route('/healthcheck', methods=['GET'])
def check():
    return jsonify({"Success": "Server up!"}), 200

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)


