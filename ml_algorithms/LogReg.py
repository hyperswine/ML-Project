# -*- coding: utf-8 -*-
"""
Logistic Regression
"""



from auxiliary.data_clean2 import clean_data
import pandas as pd
import numpy as np
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn.tree import DecisionTreeClassifier, plot_tree
from feature_selection import y_classify_five, y_classify
from scipy import optimize

# Load up dataset 1: gsmarena
data = pd.read_csv('dataset/GSMArena_dataset_2020.csv', index_col=0)

data_features = data[["oem", "launch_announced", "launch_status", "body_dimensions", "display_size", "comms_wlan", "comms_usb",
                "features_sensors", "platform_os", "platform_cpu", "platform_gpu", "memory_internal",
                "main_camera_single", "main_camera_video", "misc_price",
                "selfie_camera_video",
                "selfie_camera_single", "battery"]]

# Clean up the data into a trainable form.
df = clean_data(data_features)

y = df["misc_price"]
X = df.drop(["misc_price"], axis=1)

# convert to categorical data
lab_enc = preprocessing.LabelEncoder()
# y = lab_enc.fit_transform(y)
y = y.apply(y_classify)

# Split data into train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=0)

logReg = LogisticRegression()
logReg = logReg.fit(X_train, y_train)
y_pred = logReg.predict(X_test)
print(r"Logistic Regression score for is {}".format(accuracy_score(y_pred, y_test)))


#%%
"""
Try do custom All vs One Logistic Regression
""" 
import operator

def sigmoid(z):
    g =  1.0 / (1.0 + np.exp(-z))
    return g

def costFunction(theta, X, y, LRlambda):
    m = np.shape(y)[0]
    J = 0;
    grad = np.zeros(np.shape(theta))
    jj = -np.multiply(y,np.log(sigmoid(X*theta))) - np.multiply((1-y),np.log(1-sigmoid(X*theta)))                                                            
    J = 1/m*sum[jj] + LRlambda/(2*m)*sum(theta[2:-1]**2) #DONT regularize theta0 since its bias term
    return J

def gradFunction(theta, X, y, LRlambda):
    m = np.shape(y)[0]
    grad_tmp = 1/m * X.T *(sigmoid(X*theta)-y)
    grad = np.array([grad_tmp[1], grad_tmp[2:-1] + LRlambda/m*theta[2:-1]])
    return grad
        
    
def oneVsA(X, y, num_labels, LRlambda):
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    all_theta = np.zeros([num_labels,n+1])
    #add ones to beginning
    X = np.concatenate((np.ones([m,1]), X.to_numpy()), axis=1)
    for c in np.arange(num_labels):
        initial_theta = np.zeros([n+1,1])
        args = ( X, (y == c), LRlambda)
        result = optimize.fmin_cg(costFunction, initial_theta, fprime=gradFunction, args = args)
        all_theta[c,:] = result
    return all_theta

def findmax(l):
    max_val = max(l)
    max_idx = l.index(max_val)
    return max_idx, max_val

def predictOneVsA(all_theta, X):
    m = np.shape(X)[0]
    num_labels = np.shape(all_theta)[0]
    # p = np.zeros([np.shape(X)[1],1])
    #add ones to beginning
    X = np.concatenate((np.ones([m,1]), X.to_numpy()), axis=1)
    A = np.dot(X,all_theta.T);
    [I,V] = findmax(A);
    return I

regularization = 0.1;   #lambda
num_labels = 3; #or 5
[all_theta] = oneVsA(X, y, num_labels, regularization);