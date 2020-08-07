"""
This script is on learning a Linear Regression model.
Before writing up our own algorithms, it made sense to use the pre-existing algorithms from libraries such as sklearn. This provides us a baseline for the performance of LR on our dataset to match.

Preliminary Considerations
There were many considerations to be made. The first regarding hyper-parameters and high-dimensional data. It was important to not overthink the first few steps so considerations with bias-variance and tweaking were considered later.
"""
from sklearn.model_selection import train_test_split
import pandas as pd
from auxiliary.data_clean2 import clean_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Open Dataset
data = pd.read_csv('dataset/GSMArena_dataset_2020.csv', index_col=0)

# Some Insight
data.info()
data.head()

# NOTE: conflicting features 'main_camera_dual', 'comms_nfc', 'battery_charging', 'selfie_camera_video' resulting in many null cols.
data_features = data[
    ["oem", "launch_announced", "launch_status", "body_dimensions", "display_size", "comms_wlan", "comms_usb",
     "features_sensors", "platform_os", "platform_cpu", "platform_gpu", "memory_internal",
     "main_camera_single", "main_camera_video", "misc_price",
     "selfie_camera_video",
     "selfie_camera_single", "battery"]]

df = clean_data(data_features)

df.dropna(inplace=True)
df.reset_index(drop=True)

# Now its time to split the data

y = df["misc_price"]
X = df.drop(["key_index", "misc_price"], axis=1)

# Train & test split. 70-30 split for the preliminary split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=120, test_size=.3)

lr_model = LinearRegression()

# Batch-train LR
lr_model.fit(X_train, y_train)

# Test the model & retreive predictions
y_pred = lr_model.predict(X_test)

print(accuracy_score(y_test, y_pred))

"""
Investigating Linear Regression in more detail.
Now we investigate LR in more depth by learning our own models and tweaking parameters, normalizing 
& comparing differences.
"""


# Set up class & method defs for LR batch

class LinReg:
    """
    A streamlined linear regression
     object for batch learning.
    """

    def __init__(self):
        self.theta_pred = 0

    def fit(self, X, y):
        self.theta_pred = \
            np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        return X.dot(self.theta_pred)

    def performance(self, y_test, y_pred):
        print('Coefficients: \n', self.theta_pred)

        print('Mean squared error: %.2f'
              % mean_squared_error(y_test, y_pred))

        print('Coefficient of determination: %.2f'
              % r2_score(y_test, y_pred))


# Train LinReg
lin_reg = LinReg()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)
print(lin_reg.performance(y_test, y_pred))

# # Perform 4-fold cross-validation on the datasets
# kf_4 = KFold(n_splits=4, shuffle=True)
# kf_4.get_n_splits(X)

# for train, test in kf_4.split(X):
#     lin_reg.fit(X[train], y[train])
#     y_pred = lin_reg.predict(X[test])
#     print(lin_reg.performance(y[test], y_pred))

# # Perform 10-fold cross-validation on the datasets
# kf_10 = KFold(n_splits=10, shuffle=True)
# kf_10.get_n_splits(X)

# for train, test in kf_10.split(X):
#     lin_reg.fit(X[train], y[train])
#     y_pred = lin_reg.predict(X[test])
#     print(lin_reg.performance(y[test], y_pred))

# Regularize with L1
from sklearn import linear_model

reg = linear_model.Lasso(alpha=.1)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(reg.coef_)
print(reg.intercept_)
print(accuracy_score(y_test, y_pred))
