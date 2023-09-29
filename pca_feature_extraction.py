import os
import cv2
import numpy as np
from skimage import feature
from sklearn.decomposition import PCA
import csv

# Define the path to the main folder containing the subfolders with images
main_folder = 'lfw_processed_data'

# Define the path to the output CSV file
output_csv = 'pca_feature_lfw.csv'

# Define the LBP parameters
num_points = 24
radius = 8

# Define the number of PCA components to keep
n_components = 5

# Initialize a list to store the image labels and feature vectors
image_labels = []
feature_vectors = []

# Iterate through each subfolder and its images
for label, foldername in enumerate(os.listdir(main_folder)):
    folder_path = os.path.join(main_folder, foldername)

    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract LBP features
        lbp = feature.local_binary_pattern(gray, num_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))

        # Append the image label and LBP features to the respective lists
        image_labels.append(label)
        feature_vectors.append(hist)

# Perform PCA for feature selection
pca = PCA(n_components=n_components)
selected_features = pca.fit_transform(feature_vectors)

# Save the image labels and selected features to a CSV file
with open(output_csv, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write the header
    header = ['name'] + ['PCA_' + str(i) for i in range(n_components)]
    writer.writerow(header)

    # Write the image labels and selected features
    for label, features in zip(image_labels, selected_features):
        writer.writerow([label] + list(features))
