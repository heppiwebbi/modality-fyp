import os
import sys
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import exposure as ex
from skimage import io
from skimage import exposure
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import scipy.misc
from scipy import ndimage
import cv2
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import cv2 

### ----------------------------------------------------------------------------------------
###
### Instructions: to run this file, one needs to place the code one level outside of the 
###                 image datclear
# aset folder
### Installation instructions: 
###                
### 
### ----------------------------------------------------------------------------------------
#print("Path at terminal when executing this file")
#mac
dirname_train = os.getcwd() + "/Image Dataset/ImageCLEF2013ModalityClassificationTrainingSet/ImageCLEF2013TrainingSet"
dirname_test = os.getcwd() + "/Image Dataset/ImageCLEFTestSetGROUNDTRUTH/ImageCLEFTestSetGROUNDTRUTH"


#pc
#dirname = os.getcwd() + "\\Image Dataset\\ImageCLEF2013ModalityClassificationTrainingSet\\ImageCLEF2013TrainingSet"
#arr = os.listdir("/Users/minxi/Downloads/FIT3162/Image Dataset/ImageCLEF2013ModalityClassificationTrainingSet/ImageCLEF2013TrainingSet")

### ----------------------------------------------------------------------------------------
###
### Local Binary Pattern
###
### ----------------------------------------------------------------------------------------

# this is to perform lbp, the function is imported from skLearn
def lbpattern(img):
    #print(img)
    METHOD = 'uniform'
    plt.rcParams['font.size'] = 9
    #print(img)
    #resized_image = cv2.resize(img, (256, 256)) 

    # settings for LBP
    radius = 3
    n_points = 8 * radius

    lbp_img = local_binary_pattern(img, n_points, radius, METHOD)
    #print(lbp_img.flatten())
    #flattened = lbp_img.flatten()
    return lbp_img


### ----------------------------------------------------------------------------------------
###
### Images Preprocessing (II): Bilateral Filter (Denoising)
###
### ----------------------------------------------------------------------------------------
# this is to perform image denoise, the function is imported from openCV

def denoise(img):  
    median = cv2.medianBlur(img,3)
    bilat = cv2.bilateralFilter(img, 5, 10, 10)
    
    """ plt.figure(figsize=(12, 2.8))

    plt.subplot(131)
    plt.imshow(img, cmap=plt.cm.gray, vmin=40, vmax=220)
    plt.axis('off')
    plt.title('Original image', fontsize=20)
    plt.subplot(132)
    plt.imshow(bilat, cmap=plt.cm.gray, vmin=40, vmax=220)
    plt.axis('off')
    plt.title('Bilateral filter', fontsize=20)
    plt.subplot(133)
    plt.imshow(median, cmap=plt.cm.gray, vmin=40, vmax=220)
    plt.axis('off')
    plt.title('Median filter', fontsize=20)

    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0, left=0,
                        right=1)

    plt.show() """
    
    return bilat


### ----------------------------------------------------------------------------------------
###
### Images Preprocessing (I): Adaptive Histogram Equalisation (Enhancement)
###
### ----------------------------------------------------------------------------------------
# this is to perform image enhancement, the function is imported from openCV
def enhance_hist(img):
    # Equalize the image
    equalizedImg = cv2.equalizeHist(img)

    # Use instead Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    claheImg = clahe.apply(img)

    # Display the results
    """ plt.figure(figsize=(12,2.8))
    plt.subplot(121)
    plt.title('Original')
    plt.imshow(img, cmap=plt.cm.gray, vmin=40, vmax=220)
    plt.axis("off")

    plt.subplot(122)
    plt.title('CLAHE')
    plt.imshow(claheImg, cmap = plt.cm.gray, vmin=40, vmax=220)
    plt.axis("off")

    plt.show()   """     

    return claheImg


### -----------------------------------------------------------------------------------------------------------------------
## this is the part for reading the file and converting it into a numerical values and store it into a list
# and the count is when user wants to limit the amount of images to be read from each classes
# there is an if condition checking for DS.Store because if it is read into the array, then the program will run into problem in the later stage.
def readFile(n, limit):
    print('starting reading training image')
    imageList = []
    labelList = []
    labelFusion = []
    count = 0
    for i in os.listdir(n):
        if i != '.DS_Store':
            labelFusion.append(i)
            temp = n + "/" + i
            for j in os.listdir(temp):
                if count < limit:
                    count += 1
                    if j != '.DS_Store':
                        img = temp + "/" + j
                        #print('image name:',j)
                        image = cv2.imread(img, 0)
                        # this is to resize the image into the form of 256 x 256S
                        image = np.resize(image,(256,256))
                        imageList.append(image)
                        labelList.append(i)
            count = 0
        
    return imageList, labelList, labelFusion


### -----------------------------------------------------------------------------------------------------------------------
# this is the feature extraction part
# where image enhancement and image denoise is applied
# and lastly it will perform LBP on the pre processed image
def featureExtraction(n):
    lbpList = []
    for i in n:
        enhanced = enhance_hist(i)
        denoised = denoise(enhanced)
        lbp = lbpattern(denoised)
        lbp = lbp.flatten()
        lbpList.append(lbp)
    return lbpList


# the fusion algorithm
# what it does is will take the probabilities of all classes that a image has
# and compare it between svm and knn
# and take the prediction from whichever classifier that has probabilities of prediction
def fusion(svm_predictions_fusion, knn_predictions_fusion, labelFusion) :

    maxIndex = []
    for i in range(len(svm_predictions_fusion)):
        if format(max(svm_predictions_fusion[i]),'.64f')  > format(max(knn_predictions_fusion[i]),'.64f'):
            maxIndex.append(np.argmax(svm_predictions_fusion[i]))
        else:
            maxIndex.append(np.argmax(knn_predictions_fusion[i]))

    labelFusion = sorted(labelFusion)

    fusion = []
    for i in maxIndex:
        fusion.append(labelFusion[i])
    print(fusion)
    count = 0

    for i in range(len(fusion)):
        if fusion[i] == labelTest[i]:
            count += 1
    temp = count/len(fusion)

    return temp

#-----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
# read image for training data


    # read image for training data
    print('starting reading training set')
    imageTrain, labelTrain, labelFusion = readFile(dirname_train, 100)
    print('finished reading traiing set')



    print('starting reading testing set')
    imageTest, labelTest, labelFusion = readFile(dirname_test, 100)
    print('finished reading testing set')



    print('------------------------------------------------------------------------------------------------------')


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# this is the part where feature extraction was run
# in this code, we are running lbp
    print('start canny on train dataset')
    lbpTrain = featureExtraction(imageTrain)
    print('finished feature extraction on traning set')


    print('start canny on test dataset')
    lbpTest = featureExtraction(imageTest)
    print('finished feature extraction on testing set')



    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# this part we will be running svm and knn classifiers on the feature extracted
    print('preparing for SVM....')
    X = lbpTrain
    y = labelTrain

    test = lbpTest
    

    print('svm starts')
    #svm_model_linear = SVC(kernel = 'rbf', verbose = True, C = 1000, gamma = 500).fit(X, y) 
    svm_model_linear = SVC(probability=True).fit(X, y) 
    svm_predictions = svm_model_linear.predict(test) 
    svm_predictions_fusion = svm_model_linear.predict_proba(test)
    # model accuracy for X_test   
    accuracy = svm_model_linear.score(test, labelTest) 

    print('svm accuracy:', accuracy)
    print('svm label prediction vvv')
    print(svm_predictions)
    print('------------------------------------------------------------------------------------------------------')
  
    print('\n\n')
  

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print('preparing for KNN....')
    knn = KNeighborsClassifier(n_neighbors = 5).fit(X, y) 
    

    # creating a confusion matrix 
    knn_predictions = knn.predict(test)  
    knn_predictions_fusion = knn.predict_proba(test)
    accuracy = knn.score(test, labelTest) 
    print('knn accuracy:', accuracy)
    print('svm label prediction vvv')
    print(knn_predictions)
    print('------------------------------------------------------------------------------------------------------')
    #cm = confusion_matrix(y_test, knn_predictions)

#----------------------------------------------------------------------------------------------
# fusion
    print('\n\n')
    print('fusion result:', fusion(svm_predictions_fusion, knn_predictions_fusion, labelFusion))

    

