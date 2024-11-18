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
    X_train, y_train, paths_train = [], [], []
    X_test, y_test, paths_test = [], [], []

    for person_folder in os.listdir(main_directory):
        person_path = os.path.join(main_directory, person_folder)
        print(person_path)
        if os.path.isdir(person_path):
            try:
                X_test_main, y_test_main, paths_test_main = load_images_with_labels(person_path, use_subfolders_for_training=False)
                X_test.extend(X_test_main)
                y_test.extend(y_test_main)
                paths_test.extend(paths_test_main)
            except Exception as e:
                print(f"No images found in main folder {person_folder}: {e}")

            # Process subfolders
            for subfolder in os.listdir(person_path):
                subfolder_path = os.path.join(person_path, subfolder)
                if os.path.isdir(subfolder_path):
                    if 'colour' in subfolder.lower():
                        X_test_subfolder, y_test_subfolder, paths_test_subfolder = load_images_with_labels(subfolder_path, use_subfolders_for_training=False)
                        X_test.extend(X_test_subfolder)
                        y_test.extend(y_test_subfolder)
                        paths_test.extend(paths_test_subfolder)
                    else:
                        X_train_subfolder, y_train_subfolder, paths_train_subfolder = load_images_with_labels(subfolder_path, use_subfolders_for_training=True)
                        X_train.extend(X_train_subfolder)
                        y_train.extend(y_train_subfolder)
                        paths_train.extend(paths_train_subfolder)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    # Flatten the images for logistic regression input
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Scale the dataset to allow logistic regression to converge faster
    scaler = StandardScaler()
    X_train_flat_scaled = scaler.fit_transform(X_train_flat)
    X_test_flat_scaled = scaler.transform(X_test_flat)

    # Train logistic regression baseline
    logistic_model = LogisticRegression(max_iter=100, solver='saga')
    logistic_model.fit(X_train_flat_scaled, y_train)

    # Predict and evaluate
    y_pred_logistic = logistic_model.predict(X_test_flat_scaled)
    baseline_accuracy = accuracy_score(y_test, y_pred_logistic)

    print(f"Logistic Regression Baseline Accuracy: {baseline_accuracy * 100:.2f}%")

    if len(X_test) == 0 or len(y_test) == 0:
        print("No test data found. Ensure the main directory contains images for testing.")
    elif len(X_train) == 0 or len(y_train) == 0:
        print("No training data found. Ensure the subdirectories contain images for training.")
    else:
        model_path = 'person_combined_model.h5'
        y_train_cat = to_categorical(y_train, num_classes=21)
        y_test_cat = to_categorical(y_test, num_classes=21)
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

        # combined_model.fit(
        #     X_train,
        #     {'decoded': X_train, 'classification': y_train_cat},
        #     batch_size=64,
        #     epochs=50,
        #     validation_data=(X_test, {'decoded': X_test, 'classification': y_test_cat})
        # )

        combined_model.save(model_path)
        print("Model saved after training.")

        # Evaluate on test data
        decoded_imgs, predictions = combined_model.predict(X_test)
        predicted_labels = np.argmax(predictions, axis=1)
        
        regular_correct = 0
        regular_wrong = 0
        colour_correct = 0
        colour_wrong = 0

        for i in range(len(y_test)):
            actual = people_orientation_map[y_test[i]]
            predicted = people_orientation_map[predicted_labels[i]]
            is_colour = 'colour' in paths_test[i]
            if actual == predicted:
                if is_colour:
                    colour_correct += 1
                else:
                    regular_correct += 1
            else:
                if is_colour:
                    colour_wrong += 1
                    print(f"Colour Image Wrong: {paths_test[i]}, predicted: {predicted}, actual: {actual}")
                else:
                    regular_wrong += 1
                    print(f"Regular Image Wrong: {paths_test[i]}, predicted: {predicted}, actual: {actual}")

        print(f"Regular Correct: {regular_correct}")
        print(f"Regular Wrong: {regular_wrong}")
        print(f"Colour Correct: {colour_correct}")
        print(f"Colour Wrong: {colour_wrong}")
