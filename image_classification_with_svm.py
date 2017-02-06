import os, os.path, sys
import re
import numpy as np
import cvxopt
import cvxopt.solvers
from PIL import Image

## Read image (pgm) files ##
def read_pgm(filename):
    X=[]
    im = Image.open(filename)
    ## convert image matrix into a single array
    X = [item for sublist in np.asarray(im) for item in sublist]
    return X

## Create a dictionary with all data ##
def get_data_list():
    print('Reading input files')
    folderList = {}
    b = os.listdir(DIR)
    print('Reading files from the data folder')
    for foldername in b:
        fileList = []
        fileList.extend([name for name in os.listdir(DIR+"/"+foldername) 
            if os.path.isfile(os.path.join(DIR+"/"+foldername, name))])
        folderList[foldername] = fileList

    return folderList

## Predict probability for each test image for all classes ##
def predict(image, wbs):
    prediction_list = {}
    for foldername in wbs:
        w_temp = wbs[foldername]
        w = w_temp[0]
        b = w_temp[1]
        pred_val = np.dot(image, w) + b
        prediction_list[foldername]=pred_val
    
    return prediction_list

## Train input by calculating alpha, and using it to calculate w and b ##
def train(X, y):
    X = np.asarray(X, dtype=np.float)
    y = np.asarray(y, dtype=np.float)
    
    num_samples, num_features = X.shape

    temp_mat = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            temp_mat[i,j] = np.dot(X[i], X[j])
    
    yTrans = y.transpose()
    part = np.outer(y,yTrans)
    H = part*temp_mat
    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(np.ones(num_samples) * -1)    
    G = cvxopt.matrix(np.diag(np.ones(num_samples) * -1))
    h = cvxopt.matrix(np.zeros(num_samples))    
    A = cvxopt.matrix(  y, (1,num_samples), tc='d' )    
    b = cvxopt.matrix(0.0)
    
    ## solve QP problem using cvxopt lib ##
    cvxopt.solvers.options['show_progress'] = False
    ans = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = np.ravel(ans['x'])
    
    ## find Support vectors that have greater values ##
    sv = alpha > 1e-9
    sv = np.array(sv,dtype=bool)
    ind = np.arange(len(alpha))[sv]
    s = alpha[sv]
    new_sv=X[sv]
    sv_y=y[sv]
    
    ## Weight vector ##
    w = np.zeros(num_features)
    for n in range(len(s)):
        w += s[n] * sv_y[n] * new_sv[n]
    
    ## b ##
    b = 0
    for n in range(len(s)):
        b += sv_y[n]
        b -= np.sum(s * sv_y * temp_mat[ind[n],sv])
    b /= len(s)
    
    return w, b

## Read all files, divide in training and testing data, perform training and testing for accuracy
def svm(position, folderList):
    i=0
    wbs = {}
    print('Training data')
    for foldernam in folderList:
        currentClass = []
        otherClasses = []
        i=i+1    
        for foldername in folderList:
            if foldername is not foldernam:
                jlo = int(len(folderList[foldername])/2)   
                temp = folderList[foldername]
                ## Position is used to decide first 5 or last 5 images to b used for training, top stands for first 5
                if position is 'top':
                    training_files = temp[:jlo]
                else:
                    training_files = temp[jlo:]
                
                for afile in training_files:
                    image = read_pgm(DIR+"/"+foldername+"/"+afile) 
                    otherClasses.append(image)
            else:
                jlo = int(len(folderList[foldername])/2)   
                temp = folderList[foldername]
                if position is 'top':
                    training_files = temp[:jlo]
                else:
                    training_files = temp[jlo:]
                
                for afile in training_files:
                    image = read_pgm(DIR+"/"+foldername+"/"+afile) 
                    currentClass.append(image)
        
        y1 = np.ones(len(currentClass))    
        y2 = np.ones(len(otherClasses)) * -1
        X_train = np.vstack((currentClass, otherClasses))
        y_train = np.hstack((y1, y2))
        w, b = train(X_train, y_train)
        wbs[foldernam] = [w,b]

    ## Testing phase ##
    accuracy = 0
    sorted_pred = {}
    print('Testing data')
    for foldername in folderList:
        jlo = int(len(folderList[foldername])/2)   
        temp = folderList[foldername]
        if position is 'top':
            testing_files = temp[jlo:]
        else:
            testing_files = temp[:jlo]
        
        for aFile in testing_files:
            image = read_pgm(DIR+"/"+foldername+"/"+afile) 
            prediction_list = predict(image, wbs)
            mymax = 0
            myclass = ''
            i=0
            for key in prediction_list:
                if (i == 0):
                    mymax = prediction_list[key]
                    myclass = key
                    i=i+1
                else:
                    if(prediction_list[key] > mymax):
                        mymax = prediction_list[key]
                        myclass = key
            
            ## check if predicted value is same as actual value ##
            if myclass is foldername:
                accuracy = accuracy + 1

    total_files = float(len(folderList)*10)
    final_acc = accuracy/(total_files/2)
    return final_acc

## Main 
DIR = './data/att_faces'
folderList = get_data_list()
## Calculate accuracy by exchaning training and testing data
print('Applying SVM on data set 1')
acc1 = svm('top', folderList)
print('Accuracy for set1: ' + str(acc1*100) + '%')
print('Applying SVM on data set 1')
acc2 = svm('bottom', folderList)
print('Accuracy for set2: ' + str(acc2*100) + '%')
final_acc = ((acc1+acc2)/2.0)*100
print('Final Accuracy is: '+ str(final_acc) + '%')