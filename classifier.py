import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.utils import to_categorical

head_orientation_map = {0: 'up', 1: 'straight', 2: 'left', 3: 'right'}

def load_images_with_labels(base_directory, use_subfolders_for_training=True):
    images = []
    labels = []
    file_paths = []
    label_map = {'up': 0, 'straight': 1, 'left': 2, 'right': 3}

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
                    if len(parts) > 1:
                        orientation = parts[1]
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

def get_train_test_folders(main_directory):
    # Get all person folders, sorted to ensure consistency
    person_folders = sorted([f for f in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, f))])
    
    # First 16 folders for training (including all subfolder images)
    train_folders = person_folders[:16]
    
    # Next 4 folders for testing (only original images)
    test_folders = person_folders[16:20]
    
    print("Training folders:", train_folders)
    print("Testing folders:", test_folders)
    
    return train_folders, test_folders

if __name__ == "__main__":
    main_directory = 'faces'
    X_train, y_train, paths_train = [], [], []
    X_test, y_test, paths_test = [], [], []

    # Get specific training and testing folders
    train_folders, test_folders = get_train_test_folders(main_directory)

    # Process training folders
    for person_folder in train_folders:
        person_path = os.path.join(main_directory, person_folder)
        print(f"Processing training folder: {person_path}")
        
        # Process augmented subfolders for training
        for subfolder in os.listdir(person_path):
            subfolder_path = os.path.join(person_path, subfolder)
            if os.path.isdir(subfolder_path) and 'colour' not in subfolder.lower():
                try:
                    X_train_subfolder, y_train_subfolder, paths_train_subfolder = load_images_with_labels(
                        subfolder_path, use_subfolders_for_training=True)
                    X_train.extend(X_train_subfolder)
                    y_train.extend(y_train_subfolder)
                    paths_train.extend(paths_train_subfolder)
                except Exception as e:
                    print(f"Error processing training subfolder {subfolder_path}: {e}")

    # Process testing folders
    for person_folder in test_folders:
        person_path = os.path.join(main_directory, person_folder)
        print(f"Processing testing folder: {person_path}")
        
        # Use original images (in main folder) for testing
        try:
            X_test_main, y_test_main, paths_test_main = load_images_with_labels(
                person_path, use_subfolders_for_training=False)
            X_test.extend(X_test_main)
            y_test.extend(y_test_main)
            paths_test.extend(paths_test_main)
        except Exception as e:
            print(f"Error processing test folder {person_path}: {e}")

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    if len(X_test) == 0 or len(y_test) == 0:
        print("No test data found. Ensure the main directory contains images for testing.")
    elif len(X_train) == 0 or len(y_train) == 0:
        print("No training data found. Ensure the subdirectories contain images for training.")
    else:
        model_path = 'head_orientation_combined_model.h5'
        y_train_cat = to_categorical(y_train, num_classes=4)
        y_test_cat = to_categorical(y_test, num_classes=4)
        
        if os.path.exists(model_path):
            # Load model if exists
            print("Loading existing model and continuing training.")
            combined_model = load_model(model_path)
            combined_model.compile(
                optimizer='adam',
                loss={'decoded': 'mean_squared_error', 'classification': 'categorical_crossentropy'},
                loss_weights={'decoded': 1.0, 'classification': 0.5},
                metrics={'classification': ['accuracy']}
            )
        else:
            #create model otherwise
            print("Creating a new model.")
            combined_model = create_combined_model(input_shape=(64, 64, 3), num_classes=4)
        #train model
        # combined_model.fit(
        #     X_train,
        #     {'decoded': X_train, 'classification': y_train_cat},
        #     batch_size=64,
        #     epochs=50,
        #     validation_data=(X_test, {'decoded': X_test, 'classification': y_test_cat})
        # )

        combined_model.save(model_path)
        print("Model saved after training.")

        decoded_imgs, predictions = combined_model.predict(X_test)
        predicted_labels = np.argmax(predictions, axis=1)
        
        correct = 0
        wrong = 0
        # run through testing data and find accuracy
        for i in range(len(y_test)):
            actual = head_orientation_map[y_test[i]]
            predicted = head_orientation_map[predicted_labels[i]]
            if actual == predicted:
                correct += 1
            else:
                wrong += 1
                print(f"Wrong Prediction: {paths_test[i]}, predicted: {predicted}, actual: {actual}")

        print(f"Correct Predictions: {correct}")
        print(f"Wrong Predictions: {wrong}")
        print(f"Accuracy: {correct/(correct + wrong):.2%}")