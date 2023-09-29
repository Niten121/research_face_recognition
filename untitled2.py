import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model

# Define the subfolder containing the images
subfolder_path = 'ORL_database'

# Define the target image size for CNN models
target_size = (224, 224)  # Adjust as per the model requirements

# Define the CNN models to use
models = {
    'VGG16': VGG16(weights='imagenet', include_top=False),
    'ResNet50': ResNet50(weights='imagenet', include_top=False),
    'InceptionV3': InceptionV3(weights='imagenet', include_top=False)
}

# Initialize lists to store the features and labels
features = []
labels = []

# Iterate over each subfolder (class) in the main folder
for class_folder in os.listdir(subfolder_path):
    class_path = os.path.join(subfolder_path, class_folder)
    
    # Iterate over each image file in the class folder
    for image_file in os.listdir(class_path):
        image_path = os.path.join(class_path, image_file)
        
        # Load and preprocess the image
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        
        # Extract features using each CNN model
        for model_name, model in models.items():
            cnn_features = model.predict(image)
            cnn_features = cnn_features.flatten()  # Flatten the features
            
            # Append the features and labels
            features.append(cnn_features)
            labels.append(class_folder)  # Assuming subfolder names are the class labels

# Convert features and labels to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the classifier (Logistic Regression in this example)
classifier = LogisticRegression()

# Train the classifier
classifier.fit(X_train, y_train)

# Test the classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
