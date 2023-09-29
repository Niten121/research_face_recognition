import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
from sklearn.neighbors import KNeighborsClassifier

def extract_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def extract_features_from_folder(folder_path):
    features = []
    labels = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                label = os.path.basename(root)
                image = cv2.imread(image_path)
                keypoints, descriptors = extract_sift_features(image)
                features.append(descriptors)
                labels.append(label)
    
    return features, labels

def cluster_features(features, num_clusters):
    all_descriptors = np.concatenate(features)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(all_descriptors)
    return kmeans

def calculate_histograms(features, kmeans):
    histograms = []
    for descriptors in features:
        labels = kmeans.predict(descriptors)
        histogram, _ = np.histogram(labels, bins=range(kmeans.n_clusters + 1))
        histograms.append(histogram)
    return histograms

def save_features_to_csv(features, labels, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name'] + [f'feature_{i}' for i in range(1, features.shape[1] + 1)])
        for label, descriptor in zip(labels, features):
            writer.writerow([label] + descriptor.tolist())

# Set the path to your image folder
image_folder = 'preprocessed_ORL_database'

# Set the number of clusters for K-means
num_clusters = 20

# Set the path to save the CSV file
csv_file = 'ORL_SIFT.csv'
# Extract SIFT features from the images
features, labels = extract_features_from_folder(image_folder)

# Perform K-means clustering on the features
kmeans = cluster_features(features, num_clusters)

# Calculated histograms using the cluster centroids
histograms = calculate_histograms(features, kmeans)

# Convert features and histograms to numpy arrays
features = np.concatenate(features)
histograms = np.array(histograms)

# Save the features and labels to a CSV file
save_features_to_csv(histograms, labels, csv_file)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(histograms, labels, test_size=0.2, random_state=33)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print('fitting done')

# Predictions on the test data
y_pred = knn.predict(X_test)
print("prediction done")
print(y_pred)
print(X_test.shape)

# Accuracy
accuracy_knn = accuracy_score(y_test, y_pred)
print("KNN Accuracy:", accuracy_knn * 100)  # 91.66

# Train an SVM classifier
classifier = SVC()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}")
