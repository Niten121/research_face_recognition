import os
import cv2
import numpy as np


# Define the path of the parent directory
parent_dir = "ORL_database"
# Define the path of the output folder
output_dir = "preprocessed_ORL_database"

print("process stared kindly wait ............ ")


if not os.path.exists(output_dir):
    os.mkdir(output_dir)


for subdir_name in os.listdir(parent_dir):
    subdir_path = os.path.join(parent_dir, subdir_name)

    if os.path.isdir(subdir_path):
       
        output_subdir_path = os.path.join(output_dir, subdir_name)
        if not os.path.exists(output_subdir_path):
            os.mkdir(output_subdir_path)
        
        for filename in os.listdir(subdir_path):
            # Create the path to the file
            file_path = os.path.join(subdir_path, filename)

            # Check if the file is an image
            if file_path.endswith(".jpg") or file_path.endswith(".png"):
                # Load the image
                img = cv2.imread(file_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imshow("original",gray)
            
            
                # Remove noise using a Gaussian blur
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                blurred = cv2.GaussianBlur(blurred, (3, 3), 0)
                blurred = cv2.GaussianBlur(blurred, (3, 3), 0)
                cv2.imshow("blured",blurred)

                # Sharpen the image using a unsharp mask
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(blurred, -1, kernel)
                # sharpened = cv2.filter2D(sharpened, -1, kernel)
                cv2.imshow("sharpened",sharpened)
                
        
                # Normalize the image
                normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
                normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
                cv2.imshow("normalized",normalized)
                
                # median filter
                median = cv2.medianBlur(normalized, 3)
                cv2.imshow("median:", median)
                
                # denoising
                noise_1 = cv2.fastNlMeansDenoising(median, 2, 3.0, 7, 21)
                cv2.imshow('denoise', noise_1)
                
                noise_1 = cv2.fastNlMeansDenoising(median, 2, 3.0, 7, 21)
                cv2.imshow('denoise', noise_1)
                
            
              
                processed_image = noise_1
                # Save the processed image  
                cv2.imwrite(os.path.join(output_subdir_path, filename), processed_image)
                cv2.waitKey(1)

cv2.waitKey(0)
print("pre-processing done .... ")
print("MOVE FORWARD--------------->>>>>>>Feature Extraction")
cv2.destroyAllWindows()
