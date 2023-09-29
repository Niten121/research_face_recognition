import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np

dataset1 = pd.read_csv('ORL_mix_features.csv')
dataset2 = pd.read_csv('mix_features.csv')

combined_dataset = pd.concat([dataset1, dataset2])

le = LabelEncoder()
combined_dataset['outcome'] = le.fit_transform(combined_dataset['name'])

X = combined_dataset.drop(['name', 'outcome'], axis=1)
y = combined_dataset['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=35)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=5, random_state=0)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# SVM
svm_classifier = svm.SVC()
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Accuracy values for each classifier
classifiers = ['KNN', 'SVM']
accuracies_dataset1 = [accuracy_knn*100, accuracy_svm*100]

# Apply the same classifiers on the second dataset
X2_test = scaler.transform(dataset2.drop(['name'], axis=1))
y2_test = le.transform(dataset2['name'])

y2_pred_knn = knn.predict(X2_test)
accuracy2_knn = accuracy_score(y2_test, y2_pred_knn)

# y2_pred_rf = random_forest.predict(X2_test)
# accuracy2_rf = accuracy_score(y2_test, y2_pred_rf)

y2_pred_svm = svm_classifier.predict(X2_test)
accuracy2_svm = accuracy_score(y2_test, y2_pred_svm)

accuracies_dataset2 = [accuracy2_knn*100, accuracy2_svm*100]

# Create a DataFrame for bar plot
data = {
    'ORL': accuracies_dataset1,
    'OWN': accuracies_dataset2
}
df = pd.DataFrame(data, index=classifiers)

# Plot the comparative bar graph
ax = df.plot(kind='bar', rot=0, color=['skyblue', 'lightgreen'])
ax.set_xlabel('Classifiers')
ax.set_ylabel('Accuracy')
ax.set_title('Comparative Accuracy of Classifiers')

# Add labels to each bar
for i in range(len(classifiers)):
    plt.annotate("{:.2f}%".format(df['ORL'][i]), xy=(i, df['ORL'][i]),
                 xytext=(0, 3), textcoords='offset points', ha='right', va='bottom')

    plt.annotate("{:.2f}%".format(df['OWN'][i]), xy=(i, df['OWN'][i]),
                 xytext=(0, 3), textcoords='offset points', ha='left', va='bottom')

#  y-axis tick positions and limits
plt.yticks(np.arange(0, 101, 10))
plt.ylim([0, 100])

#  legend
plt.legend(title='Datasets', bbox_to_anchor=(1.01, 1), loc='upper left')

plt.tight_layout()
plt.show()
