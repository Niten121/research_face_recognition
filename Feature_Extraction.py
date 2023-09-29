import PIL
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor
import glob
import cv2
import os
import csv

from skimage.feature import  greycomatrix, greycoprops
import warnings
warnings.simplefilter(action='ignore',category='deprecation')
label = []

data_dir = os.path.expanduser(r'collected_data')

files = []
labels = []   
for r, d, f in os.walk(data_dir):
    for file in f:
        if '.jpg' in file:
            label = r.split('\\')[-1]
            labels.append(label)
            files.append(os.path.join(r, file))

with open('features.csv', "w", newline="") as wr:
    
    writer = csv.writer(wr)
    writer.writerow(["contrast_features", "dissimilarity_features", "homogeneity_features", "energy_features","correlation_features","name"])
    i = 0
    for f in files:
        label = f.split('\\')[-1]
        img = cv2.imread(f)
        img1 = cv2.resize(img, (400, 400))
        c = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # conversion of RGB images into Gray Scale
        gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

        
        # GLCM
        gCoMat = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
        contrast = greycoprops(gCoMat, prop='contrast')
        dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
        homogeneity = greycoprops(gCoMat, prop='homogeneity')
        energy = greycoprops(gCoMat, prop='energy')
        correlation = greycoprops(gCoMat, prop='correlation')

        contrast_features = np.reshape(np.array(contrast).ravel(), (1, len(np.array(contrast).ravel())))
        dissimilarity_features = np.reshape(np.array(dissimilarity).ravel(), (1, len(np.array(dissimilarity).ravel())))
        homogeneity_features = np.reshape(np.array(homogeneity).ravel(), (1, len(np.array(homogeneity).ravel())))
        energy_features = np.reshape(np.array(energy).ravel(), (1, len(np.array(energy).ravel())))
        correlation_features = np.reshape(np.array(correlation).ravel(), (1, len(np.array(correlation).ravel())))
        
        features = np.concatenate((contrast_features, dissimilarity_features, homogeneity_features, energy_features,correlation_features), axis=1);
        ff = features[0].tolist()
        writer.writerow(ff + [labels[i]])
        i += 1
    wr.close()
cv2.waitKey(0)
warnings.simplefilter(action='ignore',category='deprecation')
print("FEATURE EXTRACTION done .... ")
print("MOVE FORWARD--------------->>>>>>>")
cv2.destroyAllWindows()