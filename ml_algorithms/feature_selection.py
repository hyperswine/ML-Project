"""
This script allows us to gain insights on what features matter the most.
It uses a random forest classifier to do this.
"""
from auxiliary.data_clean2 import clean_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


def y_classify(y):
    if y > 700:
        return 2
    elif y >= 300 and y <= 700:
        return 1

    return 0


def y_classify_five(y):
    if y > 1000:
        return 4
    elif y > 700 and y <= 1000:
        return 3
    elif y > 450 and y <= 700:
        return 2
    elif y > 200 and y <= 450:
        return 1

    return 0


def feature_selection(df):
    """
    Output the features that are the most important in the feature dataframe
    """
    y = df["misc_price"]
    # NOTE: 3 classes default. Switch this to 'y_classify+_five' for 5 classes.
    # 3 classes seems to result in a higher performance with both classifiers
    y = y.apply(y_classify)
    X = df.drop(["key_index", "misc_price"], axis=1)
    rand_forest = RandomForestClassifier(n_estimators=500, n_jobs=-1)

    rand_forest.fit(X, y)

    for feature, score in zip(X, rand_forest.feature_importances_):
        print(feature, score)

    # use the random forest to predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=120, test_size=.3)

    rand_forest.fit(X_train, y_train)
    y_pred = rand_forest.predict(X_test)
    print("Accuracy of RF classifier", accuracy_score(y_test, y_pred))

    # use a neural net to classify
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy of Multiple Layer Perceptron", accuracy_score(y_test, y_pred))

    # k-NN with k = 1...10
    for i in range(1, 11):
        clf = KNeighborsClassifier(n_neighbors=i, weights='distance')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Accuracy of NN with k = {i}", accuracy_score(y_test, y_pred))

    # Naive Bayes
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy of Gassian Naive Bayes @ default settings", accuracy_score(y_test, y_pred))

    clf = MultinomialNB(alpha=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy of Multinomial NB @ alpha = 1 (laplace)", accuracy_score(y_test, y_pred))

    clf = ComplementNB(alpha=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy of Complement NB @ alpha = 1 (laplace)", accuracy_score(y_test, y_pred))



if __name__ == "__main__":
    data = pd.read_csv('dataset/GSMArena_dataset_2020.csv',
                       index_col=0)

    data_features = data[
        ["oem", "launch_announced", "launch_status", "body_dimensions", "display_size", "comms_wlan", "comms_usb",
         "features_sensors", "platform_os", "platform_cpu", "platform_gpu", "memory_internal",
         "main_camera_single", "main_camera_video", "misc_price",
         "selfie_camera_video",
         "selfie_camera_single", "battery"]]

    df = clean_data(data_features)
    feature_selection(df)
