# classifier.py
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
		y_train_cat = to_categorical(y_train, num_classes=4)
		y_test_cat = to_categorical(y_test, num_classes=4)

		model_path = 'head_orientation_combined_model.h5'
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

		combined_model.fit(
			X_train,
			{'decoded': X_train, 'classification': y_train_cat},
			batch_size=64,
			epochs=50,
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
			actual = head_orientation_map[y_test[i]]
			predicted = head_orientation_map[predicted_labels[i]]
			if actual == predicted:
				correct += 1
			else:
				print(f"Got image wrong: {paths_test[i]}, predicted: {predicted}, actual: {actual}")
				wrong += 1

		print(f"Total Correct: {correct}")
		print(f"Total Wrong: {wrong}")
