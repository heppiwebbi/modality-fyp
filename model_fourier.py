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
### 
### ----------------------------------------------------------------------------------------

dirname_train = os.getcwd() + "/Image Dataset/ImageCLEF2013ModalityClassificationTrainingSet/ImageCLEF2013TrainingSet"
dirname_test = os.getcwd() + "/Image Dataset/ImageCLEFTestSetGROUNDTRUTH/ImageCLEFTestSetGROUNDTRUTH"



### ----------------------------------------------------------------------------------------
###
### Images Preprocessing (II): Bilateral Filter (Denoising)
###
### ----------------------------------------------------------------------------------------

def denoise(img):  
    #img = io.imread(dirname + "/DRCT/1471-2342-2-1-6.jpg")
    #noisy = img + 0.4*img.std()*np.random.random(img.shape)
    #gauss_denoised = ndimage.gaussian_filter(noisy, 1)
    #med_denoised = ndimage.median_filter(noisy, 1)
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
### Fourier transformation
###
### ----------------------------------------------------------------------------------------
def fourier(img):
    resultimg = []
    # Load the image in gray scale
    rows, cols = img.shape

    # Transform the image to improve the speed in the fourier transform calculation
    optimalRows = cv2.getOptimalDFTSize(rows)
    optimalCols = cv2.getOptimalDFTSize(cols)
    optimalImg = np.zeros((optimalRows, optimalCols))
    optimalImg[:rows, :cols] = img
    crow, ccol = int(optimalRows / 2) , int(optimalCols / 2)

    # Calculate the discrete Fourier transform
    dft = cv2.dft(np.float32(optimalImg), flags=cv2.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dft)

    # Mask everything except the center
    mask = np.zeros((optimalRows, optimalCols, 2), np.uint8)
    mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 1
    dftShift = dftShift * mask

    # Rescale the values for visualization purposes
    magnitudeSpectrum = 20 * np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))

    # Reconstruct the image using the inverse Fourier transform
    newDft = np.fft.ifftshift(dftShift)
    result = cv2.idft(newDft)
    result = cv2.magnitude(result[:, :, 0], result[:, :, 1])
    '''
    # Display the results
    images = [optimalImg, magnitudeSpectrum, result]
    imageTitles = ['Input image', 'Magnitude Spectrum', 'Result']

    for i in range(len(images)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(imageTitles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()
    # Destroy all windows
    cv2.destroyAllWindows()'''

    return result

### ----------------------------------------------------------------------------------------
###
### Read File
###
### ----------------------------------------------------------------------------------------
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
            temp = n + "/" + i
            labelFusion.append(i)
            for j in os.listdir(temp):
                if count < limit:
                    count += 1
                    if j != '.DS_Store':
                        img = temp + "/" + j
                        #print('image name:',j)
                        image = cv2.imread(img, 0)
                        image = np.resize(image,(256,256))
                        imageList.append(image)
                        labelList.append(i)
            count = 0
        
    return imageList, labelList, labelFusion


### ----------------------------------------------------------------------------------------
###
### Feature Extraction
###
### ----------------------------------------------------------------------------------------
# this is the feature extraction part
# where image enhancement and image denoise is applied
# and lastly it will perform LBP on the pre processed image
def featureExtraction(n):
    fourierList = []
    for i in n:
        enhanced = enhance_hist(i)
        denoised = denoise(enhanced)
        ff = fourier(denoised)
        ff = ff.flatten()
        fourierList.append(ff)
    return fourierList


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
# read image for training data


    # read image for training data
    print('starting reading training set')
    imageTrain, labelTrain, labelFusion = readFile(dirname_train, 1)
    print('finished reading traiing set')


    print('starting reading testing set')
    imageTest, labelTest, labelFusion = readFile(dirname_test, 1)
    print('finished reading testing set')


    #print(labelTrain)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# this is the part where feature extraction was run
# in this code, we are running canny descriptors
    print('start fourier on train dataset')
    fourierTrain = featureExtraction(imageTrain)
    print('finished feature extraction on traning set')

    print('start fourier on testing dataset')
    fourierTest = featureExtraction(imageTest)
    print('finished feature extraction on testing set')


    print('preparing for SVM....')

    X = fourierTrain
    y = labelTrain

    test = fourierTest
    
    
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

    
