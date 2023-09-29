import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier

def process_dataset(dataset_folder):
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
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2,random_state=20)

    # Normalize the pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Reshape the data for CNN input (add channel dimension for grayscale images)
    X_train = X_train.reshape(-1, 64, 64, 1)
    X_test = X_test.reshape(-1, 64, 64, 1)

    return X_train, X_test, y_train, y_test

def train_classifier(X_train, y_train):
    # Define the CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')  # Output layer with the number of classes
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=5)

    return model

def evaluate_classifier(model, X_test, y_test):
    # Extract CNN features from the test set
    cnn_features_test = model.predict(X_test)

    # Train kNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(cnn_features_test, y_test)
    knn_predictions = knn_classifier.predict(cnn_features_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)

    # Train SVM classifier
    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(cnn_features_test, y_test)
    svm_predictions = svm_classifier.predict(cnn_features_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)

    # # Train Random Forest classifier
    # rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf_classifier.fit(cnn_features_test, y_test)
    # rf_predictions = rf_classifier.predict(cnn_features_test)
    # rf_accuracy = accuracy_score(y_test, rf_predictions)

    return knn_accuracy*100, svm_accuracy*100

def plot_accuracy_bar_graph(accuracies_dataset1, accuracies_dataset2):
    classifiers = ['KNN', 'SVM']

    data = {
        'ORL': accuracies_dataset1,
        'OWN': accuracies_dataset2
    }
    df = pd.DataFrame(data, index=classifiers)

    ax = df.plot(kind='bar', rot=0, color=['skyblue', 'lightgreen'])
    ax.set_xlabel('Classifiers')
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparative Accuracy of Classifiers',pad= '0.3')

    for i in range(len(classifiers)):
        plt.annotate("{:.2f}%".format(df['ORL'][i]), xy=(i, df['ORL'][i]),
                     xytext=(0, 3), textcoords='offset points', ha='right', va='bottom')

        plt.annotate("{:.2f}%".format(df['OWN'][i]), xy=(i, df['OWN'][i]),
                     xytext=(0, 3), textcoords='offset points', ha='left', va='bottom')

    plt.legend(title='Datasets')
    plt.ylim([0, 105])
    plt.yticks(np.arange(0, 105, 10))
    plt.tight_layout()
   
    plt.legend(title='Datasets', bbox_to_anchor=(1, 1))
    plt.show()
    

# Process and train on Dataset 1
X_train_1, X_test_1, y_train_1, y_test_1 = process_dataset('preprocessed_ORL_database')
model_1 = train_classifier(X_train_1, y_train_1)
accuracies_dataset1 = evaluate_classifier(model_1, X_test_1, y_test_1)

# Process and train on Dataset 2
X_train_2, X_test_2, y_train_2, y_test_2 = process_dataset('processed_data')
model_2 = train_classifier(X_train_2, y_train_2)
accuracies_dataset2 = evaluate_classifier(model_2, X_test_2, y_test_2)

# Plot the accuracy bar graph
plot_accuracy_bar_graph(accuracies_dataset1, accuracies_dataset2)

