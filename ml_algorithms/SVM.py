# Support Vector Machines

# Load scripts to clean and generate data
# noinspection PyUnresolvedReferences
from auxiliary.data_clean2 import clean_data
import pandas as pd
import numpy as np

data = pd.read_csv('dataset/GSMArena_dataset_2020.csv', index_col=0)

data_features = data[
    ["oem", "launch_announced", "launch_status", "body_dimensions", "display_size", "comms_wlan", "comms_usb",
     "features_sensors", "platform_os", "platform_cpu", "platform_gpu", "memory_internal",
     "main_camera_single", "main_camera_video", "misc_price",
     "selfie_camera_video",
     "selfie_camera_single", "battery"]]

# Clean up the data into a trainable form.
df = clean_data(data_features)

# Load helper functions
from feature_selection import y_classify, y_classify_five

# Now its time to split the data
from sklearn.model_selection import train_test_split

y = df["misc_price"]
X = df.drop(["key_index", "misc_price"], axis=1)
# one numeric variable = screen size.
X = X.drop(["screen_size", "scn_bdy_ratio", "clock_speed", "battery"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=120, test_size=.3)

y5 = y.apply(y_classify_five)
X_train5, X_test5, y_train5, y_test5 = train_test_split(X, y5, random_state=100, test_size=.3)

y3 = y.apply(y_classify)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y3, random_state=80, test_size=.3)

"""
Prelininary investigation into SVM performance. Here, we establish a baseline for how well an SVM should perform 
in practice.
"""

from sklearn.svm import SVC, SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, plot_roc_curve, classification_report

# NOTE: default radial basis kernel
svm_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svm_clf.fit(X_train3, y_train3)

y_pred3 = svm_clf.predict(X_test3)
print("3 class accuracy: ", accuracy_score(y_test3, y_pred3))
# plot_roc_curve(svm_clf, X_test, y_test) - Unfortunately only works for binary pipelines

svm_clf.fit(X_train5, y_train5)

y_pred5 = svm_clf.predict(X_test5)
print("5 class accuracy: ", accuracy_score(y_test5, y_pred5))

svr_clf = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=.2))
svr_clf.fit(X_train, y_train)
print("Support Vector Regression score: ", svr_clf.score(X_test, y_test))


"""
It appears that 3-class classification would work the best. Accurate SV regression seems highly unlikely.
Hence, in the next stage we will build our own multiclass SVM classifier & utilize various nonlinear kernels.
"""

class HyperSVM:
    """
    A support-vector machine with multiple kernel mappings for high
    dimensions & hinge loss. Uses a One-vs-One strategy for multiclass classification.
    """

    def __init__(self, dual=True):
        self.dual = dual
        self.svm_models = []

    def fit(self, X, y):
        """
        Fit m(m-1)/2 models in 'ovo' manner, given m features.
        NOTE: Maximum 30 features = 435 models. Assumes that X.shape[1] contains number of features.
        """
        self.svm_models = []
        for feature_i in X.features:
            # add each model
            self.svm_models.append([self.fit_model(feature_i, feature_j) for feature_j in X.features])

    def fit_model(i, j):
        """
        i - numpy series of a feature column.
        j - numpy series of another feature column.
        """
        # fit a linear svm. TODO: change this to non-linear
        pass

    def partial_lagrangian(self, L, var):
        """
        Calculates the partial lagriangian derivative with respect to var.
        Minimizes hinge loss -> maximizes dual.

        Return - A binary SVM that outputs a score for each class according to its euclidean distance from the maximum-separating-hyperplane.
        NOTE: positive score for the side of the 'positive' (first) class, negative otherwise.
        """
        pass

    def predict(self, X):
        """
        Input data into all models & retreive an output and its associated 'score'.
        The class with the highest total score is the predicted class.

        Return - 1xm list of predictions.
        """
        # append the score for each output feature
        # scores_pair are stored as "(feature 1, feature 2)": "(score 1, score 2)" mappings
        scores_pair = {}
        prediction = []
        # main loop to predict all examples
        for index, example in X.iterrows():
            for svm_mod in self.svm_models:
                # TODO: each SvmMod class should have a predict function which takes in a length m row
                # each SvmMod should also have 2 feature names to take the 2 feature values of that row
                scores_pair[svm_mod.feature1] = svm_mod.predict(example)

            # retreive the feature with the highest total score & append to prediction
            prediction.append(self.arg_max(scores_pair))
            # reset scores pair to predict next example
            scores_pair = {}

        return prediction

    def performance(self, y_test, y_pred):
        """
        Output accuracy score.
        """
        pass

    def argmax(self):
        pass


class SvmMod:
    """
    Representation of a binary, (currently) linear SVM.
    """

    def __init__(self, class1, class2):
        self.classes = (class1, class2)
        self.a_star = []


    def fit(self, X, y):
        """
        Expect - dataframe of two feature columns.
        """
        # NOTE: y is +/-1
# a* = argmax<1..n>(-1/2 * sum<i=1..n>( sum<j=1..n>( a[i]*a[j]*y[i]*y[j]*(X[i].dot(X[j]) )) + sum<i=1..n>(a[i]))
        a_star = []



    def predict(self, X):
        """
        Return - tuple containing scores for class1 & class2
        """
        pass


# Kernel function [1]

# 4-F Cross-Validation on [1]

# Performance Insights


# Kernel function [2]

# 4-F Cross-Validation on [2]

# Performance Insights


# Kernel function [3]

# 4-F Cross-Validation on [2]

# Performance Insights


# Plots & Analysis of different Kernel methods
