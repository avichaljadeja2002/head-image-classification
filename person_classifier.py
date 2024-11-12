# classifier.py
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.utils import to_categorical

people_orientation_map = {0: 'an2i', 1: 'at33', 2: 'boland', 3: 'bpm', 4: 'ch4f', 5: 'cheyer', 6: 'choon', 7: 'danieln', 8: 'glickman',9: 'karyadi',10: 'kawamura', 11: 'kk49',12: 'megak',13: 'mitchell', 14: 'night', 15: 'phoebe',16: 'saavik',17: 'steffi',18: 'sz24',19: 'tammo'}

def load_images_with_labels(base_directory, use_subfolders_for_training=True):
    images = []
    labels = []
    file_paths = []
    label_map = {'an2i': 0, 'at33': 1, 'boland': 2, 'bpm': 3, 'ch4f' : 4, 'cheyer' : 5, 'choon' : 6, 'danieln' : 7, 'glickman' : 8, 'karyadi' : 9, 'kawamura' : 10, 'kk49' : 11, 'megak' : 12, 'mitchell' : 13, 'night' : 14, 'phoebe' : 15, 'saavik' : 16, 'steffi' : 17, 'sz24' : 18, 'tammo' : 19}

    for root, dirs, files in os.walk(base_directory):
        if use_subfolders_for_training:
            if root == base_directory:
                continue
        else:
            if root != base_directory:
                continue

        for filename in files:
            if filename.endswith(('.png', '.pgm')):
                file_path = os.path.join(root, filename)
                try:
                    img = Image.open(file_path).convert('RGB')
                    img = img.resize((64, 64))
                    img_array = np.array(img) / 255.0
                    parts = filename.split('_')
                    if len(parts) > 0:
                        orientation = parts[0]
                        if orientation in label_map:
                            images.append(img_array)
                            labels.append(label_map[orientation])
                            file_paths.append(file_path)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    return np.array(images), np.array(labels), file_paths

def create_combined_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = UpSampling2D((2, 2))(encoded)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoded')(x)

    x = GlobalAveragePooling2D()(encoded)
    classification_output = Dense(num_classes, activation='softmax', name='classification')(x)

    model = Model(inputs, [decoded, classification_output])
    model.compile(
        optimizer='adam',
        loss={'decoded': 'mean_squared_error', 'classification': 'categorical_crossentropy'},
        loss_weights={'decoded': 1.0, 'classification': 0.5},
        metrics={'classification': ['accuracy']}
    )
    return model

if __name__ == "__main__":
    main_directory = 'faces'
    X_train, y_train = [], []
    X_test, y_test = [], []
    paths_train, paths_test = [], []

    for person_folder in os.listdir(main_directory):
        person_path = os.path.join(main_directory, person_folder)
        if os.path.isdir(person_path):
            X_test_person, y_test_person, paths_test_person = load_images_with_labels(person_path, use_subfolders_for_training=False)
            X_test.extend(X_test_person)
            y_test.extend(y_test_person)
            paths_test.extend(paths_test_person)

            X_train_person, y_train_person, paths_train_person = load_images_with_labels(person_path, use_subfolders_for_training=True)
            X_train.extend(X_train_person)
            y_train.extend(y_train_person)
            paths_train.extend(paths_train_person)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    if len(X_test) == 0 or len(y_test) == 0:
        print("No test data found. Ensure the main directory contains images for testing.")
    elif len(X_train) == 0 or len(y_train) == 0:
        print("No training data found. Ensure the subdirectories contain images for training.")
    else:
        y_train_cat = to_categorical(y_train, num_classes=21)
        y_test_cat = to_categorical(y_test, num_classes=21)

        model_path = 'person_combined_model.h5'
        if os.path.exists(model_path):
            print("Loading existing model and continuing training.")
            combined_model = load_model(model_path)
            combined_model.compile(
                optimizer='adam',
                loss={'decoded': 'mean_squared_error', 'classification': 'categorical_crossentropy'},
                loss_weights={'decoded': 1.0, 'classification': 0.5},
                metrics={'classification': ['accuracy']}
            )
        else:
            print("Creating a new model.")
            combined_model = create_combined_model(input_shape=(64, 64, 3), num_classes=21)

        combined_model.fit(
            X_train,
            {'decoded': X_train, 'classification': y_train_cat},
            batch_size=64,
            epochs=10,
            validation_data=(X_test, {'decoded': X_test, 'classification': y_test_cat})
        )

        combined_model.save(model_path)
        print("Model saved after training.")

        # Evaluate on test data
        decoded_imgs, predictions = combined_model.predict(X_test)
        predicted_labels = np.argmax(predictions, axis=1)

        correct = 0
        wrong = 0
        for i in range(len(y_test)):
            actual = people_orientation_map[y_test[i]]
            predicted = people_orientation_map[predicted_labels[i]]
            if actual == predicted:
                correct += 1
            else:
                print(f"Got image wrong: {paths_test[i]}, predicted: {predicted}, actual: {actual}")
                wrong += 1

        print(f"Total Correct: {correct}")
        print(f"Total Wrong: {wrong}")
