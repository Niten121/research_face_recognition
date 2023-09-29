import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.simplefilter('ignore')

# Read the dataset
dataset = pd.read_csv('my_SIFT.csv')
print(dataset)
print("shape:", dataset.shape)

print(dataset.info())
print(dataset.isnull().sum())

# Encode labels
le = LabelEncoder()
lable = le.fit_transform(dataset.name)
print(lable)

dataset["outcome"] = lable
print(dataset)

x = dataset.drop(['name', 'outcome'], axis='columns')
y = dataset.outcome

print(x)
print(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=27)

# Standardize features by removing the mean and scaling to unit variance.
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print('x.shape, x_test.shape, x_train.shape')
print(x.shape, x_test.shape, x_train.shape)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
print('fitting done')

# Predictions on the test data
y_pred = knn.predict(x_test)
print("prediction done")
print(y_pred)
print(x_test.shape)

# Confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Accuracy
accuracy_knn = accuracy_score(y_test, y_pred)
print("KNN Accuracy:", accuracy_knn * 100)  # 91.66

# # Random Forest
# model = RandomForestClassifier(n_estimators=5, random_state=0)

# # Train the model
# model.fit(x_train, y_train)
# print("fitting done")

# # Test the model
# y_pred = model.predict(x_test)
# print("prediction done")

# # Confusion matrix
# confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
# print(confusion_matrix)

# # Accuracy
# accuracy_rf = accuracy_score(y_test, y_pred)
# print("Random Forest Accuracy: {:.2f}%".format(accuracy_rf * 100))  # 87.5

# SVM
classifier = svm.SVC()

# Train the classifier
classifier.fit(x_train, y_train)
print("fitting done")

# Predictions on the test data
y_pred = classifier.predict(x_test)
print("prediction done")

# Confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Accuracy
accuracy_svm = accuracy_score(y_test, y_pred)
print("SVM Accuracy:", accuracy_svm * 100)  # 94.4

# Accuracy values for each classifier
classifiers = ['KNN', 'SVM']
accuracies = [accuracy_knn * 100, accuracy_svm * 100]

# Define colors for each bar
colors = ['skyblue', 'lightgreen', 'lightcoral']

# Create a bar plot with colored bars
plt.bar(classifiers, accuracies, color=colors)
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Classifiers')
plt.ylim([0, 100])

# Add labels to each bar
for i, v in enumerate(accuracies):
    plt.text(i, v + 1, "{:.2f}%".format(v), ha='center', color='black')

# Display the plot
plt.show()
