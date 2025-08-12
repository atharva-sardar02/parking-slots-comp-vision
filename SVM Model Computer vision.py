import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Define constants
IMAGE_SIZE = (64, 64)
BLOCK_SIZE = (16, 16)
CELL_SIZE = (8, 8)


# Load images and labels
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        print(f"Processing image: {img_path}")

        print(f"Original Image Shape: {img.shape}")  # Add this line for debugging

        try:
            img = cv2.resize(img, IMAGE_SIZE)
            print(f"Resized Image Shape: {img.shape}")  # Add this line for debugging

            hog_features = hog(img, pixels_per_cell=CELL_SIZE, cells_per_block=BLOCK_SIZE, block_norm='L2-Hys',
                               visualize=False)
            images.append(hog_features)
            labels.append(1 if folder == 'full' else 0)  # 1 for full, 0 for free
        except Exception as e:
            print(f"Error processing image: {img_path}")
            print(f"Exception: {e}")

    return np.array(images), np.array(labels)


full_images, full_labels = load_images_from_folder('full')
free_images, free_labels = load_images_from_folder('free')

# Combine data
X = np.concatenate([full_images, free_images], axis=0)
y = np.concatenate([full_labels, free_labels], axis=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train SVM
svm_clf = SVC()
svm_clf.fit(X_train, y_train)

# Predictions on the test set
y_pred = svm_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
