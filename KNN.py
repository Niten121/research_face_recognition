from sklearn.preprocessing import LabelEncoder
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

import warnings
warnings.simplefilter('ignore')


dataset = pd.read_csv('features.csv')
print(dataset)
print("shape", dataset.shape)



print(dataset.info())
print(dataset.isnull().sum())

# # Encode labels
le = LabelEncoder()
lable = le.fit_transform(dataset.name)
print(lable)

dataset["outcome"]=lable
print(dataset)


print("outcome0")
print(dataset[dataset.outcome==0].head())
print("outcome1")
print(dataset[dataset.outcome==1].head())

x=dataset.drop(['name',"outcome"],axis='columns')
y=dataset.outcome

print(x)
print(y)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2)

# Standardize features by removing the mean and scaling to unit variance.
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print('x.shape,x_test.shape,x_train.shape')
print(x.shape,x_test.shape,x_train.shape)


knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(x_train, y_train)
print('fitting done')


  # Make predictions on the test data
y_pred = knn.predict(x_test)
print("prediction done")
print(y_pred)
print(x_test.shape)
# y_pred1 = knn.predict([[8.28999373433584,1.26379072681704,0.626957509413158,0.0549983452273349,0.997230702337894]])
# print(y_pred1)
# y_pred2 = knn.predict([[6.47904761904762,1.21497493734336,0.630877660280778,0.0466044425880655,0.99852515732397]])
# print(y_pred2)
# y_pred3 = knn.predict([[3.83367794486216,1.14102130325815,0.602866118007803,0.0514995634431029,0.996698362742892]])
# print(y_pred3)
# confusion matrics
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)
# # accuracy 
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# accuracy1 = metrics.accuracy_score(y_test, y_pred)
# print(accuracy1)

# import seaborn as sns
# import matplotlib.pyplot as plt

# p=sns.pairplot(dataset, hue = 'outcome')
# # print(p)
# plt.figure(figsize=(5, 7))
# # ax = sns.distplot(dataset['outcome'], hist=False, color="r", label="Actual Value")
# # sns.distplot(y_pred, hist=False, color="b", label="Predicted Values", ax=ax)

# plt.plot(y_pred2, y_pred1)
# plt.title('Actual vs Precited value for outcome')
# plt.show()
# plt.close()


# cam = cv2.VideoCapture(0)
# classifier = cv2.CascadeClassifier('C:/Users/sethy/PycharmProjects/faceRecognition/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
# while True:
#     ret, fr = cam.read()
#     if ret == True:
#             gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
#             face_coordinates = classifier.detectMultiScale(gray, 1.3, 5)

#             for (x, y, w, h) in face_coordinates:
#                 fc = fr[y:y + h, x:x + w, :]
#                 r = cv2.resize(fc, (100, 50)).flatten().reshape(1,-1)
# # #             # GLCM
#                 gCoMat = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
#                 contrast = greycoprops(gCoMat, prop='contrast')
#                 dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
#                 homogeneity = greycoprops(gCoMat, prop='homogeneity')
#                 energy = greycoprops(gCoMat, prop='energy')
#                 correlation = greycoprops(gCoMat, prop='correlation')

#                 contrast_features = np.reshape(np.array(contrast).ravel(), (1, len(np.array(contrast).ravel())))
#                 dissimilarity_features = np.reshape(np.array(dissimilarity).ravel(), (1, len(np.array(dissimilarity).ravel())))
#                 homogeneity_features = np.reshape(np.array(homogeneity).ravel(), (1, len(np.array(homogeneity).ravel())))
#                 energy_features = np.reshape(np.array(energy).ravel(), (1, len(np.array(energy).ravel())))
#                 correlation_features = np.reshape(np.array(correlation).ravel(), (1, len(np.array(correlation).ravel())))
            
#                 features = np.concatenate((contrast_features, dissimilarity_features, homogeneity_features, energy_features,correlation_features), axis=1)
                
#                 cv2.putText(lable, y_pred[0].all, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#                 cv2.rectangle(features, (x, y), (x + w, y + w), (0, 0, 255), 2)
#                 cv2.imshow('livetime face recognition', fr)
                # # Calculate the accuracy of the model


# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy:', accuracy)

# # classifier = cv2.CascadeClassifier('C:/Users/sethy/PycharmProjects/faceRecognition/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
# # # # # Face Recognition using KNN
# # cam = cv2.VideoCapture(0)
# # while True:
# #     ret, fr = cam.read()
# #     if ret == True:
# #         gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
# #         face_coordinates = classifier.detectMultiScale(gray, 1.3, 5)
# #         for (x, y, w, h) in face_coordinates:
# #             fc = fr[y:y + h, x:x + w, :]
# #             r = cv2.resize(fc, (100, 50)).flatten().reshape(1,-1)
# # # #             # GLCM
# # #             gCoMat = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
# # #             contrast = greycoprops(gCoMat, prop='contrast')
# # #             dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
# # #             homogeneity = greycoprops(gCoMat, prop='homogeneity')
# # #             energy = greycoprops(gCoMat, prop='energy')
# # #             correlation = greycoprops(gCoMat, prop='correlation')

# # #             contrast_features = np.reshape(np.array(contrast).ravel(), (1, len(np.array(contrast).ravel())))
# # #             dissimilarity_features = np.reshape(np.array(dissimilarity).ravel(), (1, len(np.array(dissimilarity).ravel())))
# # #             homogeneity_features = np.reshape(np.array(homogeneity).ravel(), (1, len(np.array(homogeneity).ravel())))
# # #             energy_features = np.reshape(np.array(energy).ravel(), (1, len(np.array(energy).ravel())))
# # #             correlation_features = np.reshape(np.array(correlation).ravel(), (1, len(np.array(correlation).ravel())))
            
# # #             features = np.concatenate((contrast_features, dissimilarity_features, homogeneity_features, energy_features,correlation_features), axis=1)
# # #             # if r == 
# # #             text = knn.predict(features)
            
# # #             if text == True:
# # #                 cv2.putText(features, text[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
# # #                 cv2.rectangle(features, (x, y), (x + w, y + w), (0, 0, 255), 2)
# # #                 cv2.imshow('livetime face recognition', fr)
# # # #             if cv2.waitKey(1) == ord('q'):
# # # #                     break
# # # #             # else:
# # #             #     cv2.putText(fr, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
# # #             #     cv2.rectangle(fr, (x, y), (x + w, y + w), (0, 0, 255), 2)
# # #             #     cv2.imshow('livetime face recognition', fr)
# #             cv2.imshow('livetime face recognition', fr)
# #             if cv2.waitKey == ord('q'):  
# #                     break
# # #         # else:
# # #             #     print("error")
# # #             #     break



# # # Release the webcam and close the window
# cam.release()
print("execution complete ......THANKU")
cv2.destroyAllWindows()