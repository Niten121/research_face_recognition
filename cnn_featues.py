import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

dataset_folder = 'preprocessed_ORL_database'

# Load images and labels
images = []
labels = []

for subfolder in os.listdir(dataset_folder):
    subfolder_path = os.path.join(dataset_folder, subfolder)
    if os.path.isdir(subfolder_path):
        label = subfolder
        for image_name in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64, 64))  # Resize the image to your desired input size
            images.append(image)
            labels.append(label)

# Convert the lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Encode labels as integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=20)

# Normalize the pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape the data for CNN input (add channel dimension for grayscale images)
X_train = X_train.reshape(-1, 64, 64, 1)
X_test = X_test.reshape(-1, 64, 64, 1)


# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(labels)), activation='softmax')  # Output layer with the number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=5)
# # Evaluate the model on the test set
# loss, accuracy = model.evaluate(X_test, y_test)
# print("Test accuracy:", accuracy)

# Extract CNN features from the training set
cnn_features_train = model.predict(X_train)

# Train kNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(cnn_features_train, y_train)

# Extract CNN features from the test set
cnn_features_test = model.predict(X_test)

# Use kNN to classify test samples
knn_predictions = knn_classifier.predict(cnn_features_test)

# Calculate accuracy
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("Accuracy of kNN on test set:", knn_accuracy*100)

from sklearn.svm import SVC

# Train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(cnn_features_test, y_test)

# Use SVM to classify test samples
svm_predictions = svm_classifier.predict(cnn_features_test)

# Calculate accuracy
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("Accuracy of SVM on test set:", svm_accuracy*100)


# # Train Random Forest classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(cnn_features_test, y_test)

# # Use Random Forest to classify test samples
# rf_predictions = rf_classifier.predict(cnn_features_test)

# # Calculate accuracy
# rf_accuracy = accuracy_score(y_test, rf_predictions)
# print("Accuracy of Random Forest on test set:", rf_accuracy*100)
