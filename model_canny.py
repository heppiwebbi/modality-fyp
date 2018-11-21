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
###                 image dataset folder
### 
### ----------------------------------------------------------------------------------------
 
dirname_train = os.getcwd() + "/Image Dataset/ImageCLEF2013ModalityClassificationTrainingSet/ImageCLEF2013TrainingSet"
dirname_test = os.getcwd() + "/Image Dataset/ImageCLEFTestSetGROUNDTRUTH/ImageCLEFTestSetGROUNDTRUTH"



### ----------------------------------------------------------------------------------------
###
### Images Preprocessing (II): Bilateral Filter (Denoising)
###
### ----------------------------------------------------------------------------------------
# this is to perform image denoise, the function is imported from openCV

def denoise(img):  
    median = cv2.medianBlur(img,3)
    bilat = cv2.bilateralFilter(img, 5, 10, 10)
    
    # Uncomment to show plot result
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


### ----------------------------------------------------------------------------------------
###
### Canny Edge
###
### ----------------------------------------------------------------------------------------
def cannyEdge(img):
    # Apply the Canny edge detection algorithm with the initial threshold values
    threshold1 = 100
    threshold2 = 200
    edges = cv2.Canny(img, threshold1, threshold2)

    return edges

### ----------------------------------------------------------------------------------------
###
### Read File
###
### ----------------------------------------------------------------------------------------
## this is the part for reading the file and converting it into a numerical values and store it into a list
# and the count is when user wants to limit the amount of images to be read from each classes
# there is an if condition checking for DS.Store because if it is read into the array, then the program will run into problem in the later stage.
def readFile(n,limit):
    print('starting reading image')
    imageTrain = []
    labelTrain = []
    labelFusion = []
    count = 0
    for i in os.listdir(n):
        if i != '.DS_Store':
            labelFusion.append(i)
            #print('class name:', i)
            temp = n + "/" + i
            for j in os.listdir(temp):
                if count < limit:
                    count += 1
                    if j != '.DS_Store':
                        img = temp + "/" + j
                        #print('image name:',j)
                        image = cv2.imread(img, 0)
                        image = np.resize(image,(256,256))
                        imageTrain.append(image)
                        labelTrain.append(i)
            count = 0
        
    return imageTrain, labelTrain, labelFusion


### ----------------------------------------------------------------------------------------
###
### Feature Extraction
###
### ----------------------------------------------------------------------------------------
# this is the feature extraction part
# where image enhancement and image denoise is applied
# and lastly it will perform LBP on the pre processed image
def featureExtraction(n):
    cannyTrain = []
    for i in n:
        enhanced = enhance_hist(i)
        denoised = denoise(enhanced)
        canny = cannyEdge(denoised)

        canny = canny.flatten()
        cannyTrain.append(canny)

    return cannyTrain


### ----------------------------------------------------------------------------------------
###
### Fusion
###
### ----------------------------------------------------------------------------------------
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


### ----------------------------------------------------------------------------------------
###
### Main
###
### ----------------------------------------------------------------------------------------
if __name__ == "__main__":

    numOfIMG = int(sys.argv[1])

    # read image for training data
    print('starting reading training set')
    imageTrain, labelTrain, labelFusion = readFile(dirname_train, numOfIMG)
    print('finished reading traiing set')



    print('starting reading testing set')
    imageTest, labelTest, labelFusion = readFile(dirname_test, numOfIMG)
    print('finished reading testing set')



    print('------------------------------------------------------------------------------------------------------')



#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# this is the part where feature extraction was run
# in this code, we are running canny descriptors
    print('start canny on train dataset')
    cannyTrain = featureExtraction(imageTrain)
    print('finished feature extraction on traning set')

    print('start canny on test dataset')
    cannyTest = featureExtraction(imageTest)
    print('finished feature extraction on testing set')


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# this part we will be running svm and knn classifiers on the feature extracted

    X = cannyTrain
    y = labelTrain

    test = cannyTest


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

    
