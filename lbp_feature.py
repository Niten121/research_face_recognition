import numpy as np
from skimage.filters import gabor
import cv2
import os
import csv
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

heading_names = ['feat_lbp','lbphist_features1','lbphist_features2','lbpprob_features','LBP Energy', 'LBP Entropy', 'name']

label = []
featLength = 2+5+2
trainFeats = np.zeros((700,featLength))

data_dir=os.path.expanduser(r'processed_data')


files=[]
labels=[]
for r,d,f in os.walk(data_dir):
    for file in f:
        if '.jpg' in file:
            label=r.split('\\')[-1]
            labels.append(label)
            files.append(os.path.join(r,file))
with open('lbp_features.csv', "w", newline="") as wr:
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
        
        # LBP
        feat_lbp = local_binary_pattern(gray,5,2,'uniform')
        lbp_hist,_ = np.histogram(feat_lbp,2)
        lbp_hist = np.array(lbp_hist,dtype=float)
        lbp_prob = np.divide(lbp_hist,np.sum(lbp_hist))
        lbp_energy = np.nansum(lbp_prob**2)
        lbp_entropy = -np.nansum(np.multiply(lbp_prob,np.log2(lbp_prob)))  
        
        lbphist_features = np.reshape(np.array(lbp_hist).ravel(),(1,len(np.array(lbp_hist).ravel())))
        lbpprob_features=np.reshape(np.array(lbp_prob).ravel(),(1,len(np.array(lbp_prob).ravel())))
        lbpenrgy_features=np.reshape(np.array(lbp_energy).ravel(),(1,len(np.array(lbp_energy).ravel())))
        lbpento_features=np.reshape(np.array(lbp_entropy).ravel(),(1,len(np.array(lbp_entropy).ravel())))

        
        features=np.concatenate((lbphist_features,lbpprob_features,lbpenrgy_features,lbpento_features),axis=1);
        ff=features[0].tolist()     
        writer.writerow(ff+[labels[i]])
        i+=1
    wr.close()

