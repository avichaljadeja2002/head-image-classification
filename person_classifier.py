import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.utils import to_categorical

people_orientation_map = {0: 'an2i', 1: 'at33', 2: 'boland', 3: 'bpm', 4: 'ch4f', 5: 'cheyer', 6: 'choon', 7: 'danieln', 8: 'glickman',9: 'karyadi',10: 'kawamura', 11: 'kk49',12: 'megak',13: 'mitchell', 14: 'night', 15: 'phoebe',16: 'saavik',17: 'steffi',18: 'sz24',19: 'tammo'}

# Load images and labels from directory
def load_images_with_labels(base_directory):
    images = []
    labels = []
    processed_counts = {}
    label_map = {'an2i': 0, 'at33': 1, 'boland': 2, 'bpm': 3, 'ch4f' : 4, 'cheyer' : 5, 'choon' : 6, 'danieln' : 7, 'glickman' : 8, 'karyadi' : 9, 'kawamura' : 10, 'kk49' : 11, 'megak' : 12, 'mitchell' : 13, 'night' : 14, 'phoebe' : 15, 'saavik' : 16, 'steffi' : 17, 'sz24' : 18, 'tammo' : 19}
    
    for root, dirs, files in os.walk(base_directory):
        folder_name = os.path.basename(root)
        processed_counts[folder_name] = {'total': 0, 'processed': 0, 'skipped': 0}
        
        for filename in files:
            if filename.endswith(('.png', '.pgm')):
                file_path = os.path.join(root, filename)
                try:
                    processed_counts[folder_name]['total'] += 1
                    
                    # Load and preprocess image
                    img = Image.open(file_path).convert('RGB')
                    img = img.resize((64, 64))
                    img_array = np.array(img) / 255.0
                    
                    # Parse label from filename
                    parts = filename.split('_')
                    if len(parts) > 1:
                        orientation = parts[0]
                        if orientation in label_map:
                            images.append(img_array)
                            labels.append(label_map[orientation])
                            processed_counts[folder_name]['processed'] += 1
                    else:
                        processed_counts[folder_name]['skipped'] += 1
                        print(f"Skipping file with missing orientation: {filename} in {folder_name}")
                        
                except Exception as e:
                    processed_counts[folder_name]['skipped'] += 1
                    print(f"Error processing {filename} in {folder_name}: {e}")
    
    return np.array(images), np.array(labels)

# Define the combined model with autoencoder and classifier
def create_combined_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder to reconstruct the image
    x = UpSampling2D((2, 2))(encoded)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoded')(x)

    # Classification branch for orientation prediction
    x = GlobalAveragePooling2D()(encoded)
    x = Dense(64, activation='relu')(x)  # Added Dense layer
    classification_output = Dense(num_classes, activation='softmax', name='classification')(x)

    # Build model
    model = Model(inputs, [decoded, classification_output])
    model.compile(
        optimizer='adam',
        loss={
            'decoded': 'mean_squared_error',
            'classification': 'categorical_crossentropy'
        },
        loss_weights={
            'decoded': 0.5,
            'classification': 1.0
        },
        metrics={
            'classification': ['accuracy']
        }
    )

    return model

# Main script to load data, train and evaluate the model
directory = 'faces'  # specify data directory
images, labels = load_images_with_labels(directory)

# Ensure we have data to proceed
if len(images) == 0 or len(labels) == 0:
    print("No data found. Ensure the directory is correct and files match the expected format.")
else:
    # Train/test split and categorical encoding of labels
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    y_train_cat = to_categorical(y_train, num_classes=20)
    y_test_cat = to_categorical(y_test, num_classes=20)
    model_path = 'person_combined_model.h5'

    # Instantiate and train the model
    if os.path.exists(model_path):
        print("Loading existing model.")
        combined_model = load_model(model_path)
    else:
        print("Creating and training model.")
        combined_model = create_combined_model(input_shape=(64, 64, 3), num_classes=4)
        combined_model.fit(
            X_train,
            {'decoded': X_train, 'classification': y_train_cat},
            batch_size=64,
            epochs=50,
            validation_data=(X_test, {'decoded': X_test, 'classification': y_test_cat})
        )
        combined_model.save(model_path)
        print("Model saved after training.")
    # Make predictions on test data
    decoded_imgs, predictions = combined_model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    # Map numeric labels back to string labels for readability
    correct, wrong = 0, 0

    # Print predictions for comparison
    for i in range(len(y_test)):
        actual = orientation_map[y_test[i]]
        predicted = orientation_map[predicted_labels[i]]
        print(f"Actual: {actual}, Predicted: {predicted}")
        if actual == predicted:
            correct += 1
        else:
            wrong += 1
    print(f"Total Correct: {correct}")
    print(f"Total Wrong: {wrong}")
     