# classifier.py
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.utils import to_categorical

sunglasses_orientation_map = {0: 'no_sunglasses', 1: 'sunglasses'}

def load_images_with_labels(base_directory, use_subfolders_for_training=True):
    images = []
    labels = []
    file_paths = []
    label_map = {'no_sunglasses': 0, 'sunglasses': 1}

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
                    if "sunglasses" in filename:
                        label = label_map['sunglasses']
                    else:
                        label = label_map['no_sunglasses']
                    images.append(img_array)
                    labels.append(label)
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

    if len(X_test) == 0 or len(y_test) == 0:
        print("No test data found. Ensure the main directory contains images for testing.")
    elif len(X_train) == 0 or len(y_train) == 0:
        print("No training data found. Ensure the subdirectories contain images for training.")
    else:
        model_path = 'sunglasses_model.h5'
        y_train_cat = to_categorical(y_train, num_classes=4)
        y_test_cat = to_categorical(y_test, num_classes=4)
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
            combined_model = create_combined_model(input_shape=(64, 64, 3), num_classes=4)

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
            actual = sunglasses_orientation_map[y_test[i]]
            predicted = sunglasses_orientation_map[predicted_labels[i]]
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
