import numpy as np
from skimage.filters import gabor
import cv2
import os
import csv
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

heading_names = [ 'Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'Gabor Energy', 'Gabor Entropy', 'name']

label = []
featLength = 5+2
trainFeats = np.zeros((700,featLength))

data_dir=os.path.expanduser(r'preprocessed_ORL_database')

files=[]
labels=[]
for r,d,f in os.walk(data_dir):
    for file in f:
        if '.jpg' in file:
            label=r.split('\\')[-1]
            labels.append(label)
            files.append(os.path.join(r,file))
with open('ORL_gabor_glcm.csv', "w", newline="") as wr:
    writer = csv.writer(wr)
    writer.writerow(heading_names) # Write the heading names to the CSV file
    i=0
    for f in files:
        label=f.split('\\')[-1]
        img=cv2.imread(f)  
        img1= cv2.resize(img, (400,400))
        c=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        #conversion of RGB images into Gray Scale
        gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
      


        # GLCM 
        gCoMat = graycomatrix(gray, [1], [0],256,symmetric=True, normed=True)
        contrast = graycoprops(gCoMat, prop='contrast')
        dissimilarity = graycoprops(gCoMat, prop='dissimilarity')
        homogeneity = graycoprops(gCoMat, prop='homogeneity')    
        energy = graycoprops(gCoMat, prop='energy')
        correlation = graycoprops(gCoMat, prop='correlation')    
        
        contrast_features = np.reshape(np.array(contrast).ravel(),(1,len(np.array(contrast).ravel())))
        dissimilarity_features=np.reshape(np.array(dissimilarity).ravel(),(1,len(np.array(dissimilarity).ravel())))
        homogeneity_features=np.reshape(np.array(homogeneity).ravel(),(1,len(np.array(homogeneity).ravel())))
        energy_features=np.reshape(np.array(energy).ravel(),(1,len(np.array(energy).ravel())))
        correlation_features=np.reshape(np.array(correlation).ravel(),(1,len(np.array(correlation).ravel())))
    

        # Gabor filter
        gaborFilt_real,gaborFilt_imag = gabor(gray,frequency=0.6)
        gaborFilt = (gaborFilt_real*2+gaborFilt_imag*2)//2
        gabor_hist,_ = np.histogram(gaborFilt,8)
        gabor_hist = np.array(gabor_hist,dtype=float)
        gabor_prob = np.divide(gabor_hist,np.sum(gabor_hist))
        gabor_energy = np.nansum(gabor_prob**2)
        gabor_entropy = -np.nansum(np.multiply(gabor_prob,np.log2(gabor_prob)))
        
        #gabor_hist_features = np.reshape(np.array(gabor_hist).ravel(),(1,len(np.array(gabor_hist).ravel())))
        #gabor_prob_features=np.reshape(np.array(gabor_prob).ravel(),(1,len(np.array(gabor_prob).ravel())))
        gabor_ener_features=np.reshape(np.array(gabor_energy).ravel(),(1,len(np.array(gabor_energy).ravel())))
        gabor_entr_features=np.reshape(np.array(gabor_entropy).ravel(),(1,len(np.array(gabor_entropy).ravel())))
       
        
        features=np.concatenate((contrast_features,dissimilarity_features,homogeneity_features,energy_features,correlation_features,gabor_ener_features,gabor_entr_features),axis=1);
        ff=features[0].tolist()     
        writer.writerow(ff+[labels[i]])
        i+=1
    wr.close()

