import PIL
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor
import glob
import cv2
import os
import csv

from skimage.feature import  graycomatrix, graycoprops
import warnings
warnings.simplefilter(action='ignore',category='deprecation')
label = []

data_dir = os.path.expanduser(r'preprocessed_ORL_database')

files = []
labels = []   
for r, d, f in os.walk(data_dir):
    for file in f:
        if '.jpg' in file:
            label = r.split('\\')[-1]
            labels.append(label)
            files.append(os.path.join(r, file))

with open('ORL_glcm.csv', "w", newline="") as wr:
    
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

        # LBP
        # feat_lbp = local_binary_pattern(gray, 5, 2, 'uniform')
        # lbp_hist, _ = np.histogram(feat_lbp, 8)
        # lbp_hist = np.array(lbp_hist, dtype=float)
        # lbp_prob = np.divide(lbp_hist, np.sum(lbp_hist))
        # lbp_energy = np.nansum(lbp_prob ** 2)
        # lbp_entropy = -np.nansum(np.multiply(lbp_prob, np.log2(lbp_prob)))

        # lbphist_features = np.reshape(np.array(lbp_hist).ravel(), (1, len(np.array(lbp_hist).ravel())))
        # lbpprob_features = np.reshape(np.array(lbp_prob).ravel(), (1, len(np.array(lbp_prob).ravel())))
        # lbpenrgy_features = np.reshape(np.array(lbp_energy).ravel(), (1, len(np.array(lbp_energy).ravel())))
        # lbpento_features = np.reshape(np.array(lbp_entropy).ravel(), (1, len(np.array(lbp_entropy).ravel())))

        # GLCM
        gCoMat = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
        contrast = graycoprops(gCoMat, prop='contrast')
        dissimilarity = graycoprops(gCoMat, prop='dissimilarity')
        homogeneity = graycoprops(gCoMat, prop='homogeneity')
        energy = graycoprops(gCoMat, prop='energy')
        correlation = graycoprops(gCoMat, prop='correlation')

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
print("glcm_FEATURE EXTRACTION done .... ")
print("MOVE FORWARD--------------->>>>>>>")
cv2.destroyAllWindows()