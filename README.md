## Predicting the value of Mobile devices from their specifications

#### How to run the scripts
The 'GSMArena_dataset_2020' is around 10MB which may be too large for 'give'. Hence to run the scripts, one must download
the dataset from our team's [onedrive folder](https://unsw-my.sharepoint.com/:f:/g/personal/z5258237_ad_unsw_edu_au/EmcRr_EP6KRJlDrbOFre6ZQBBrJpezdJXhIAb0guwC7Pgw?e=lyzaRz).
NOTE: this assumes you are using a UNSW account to access onedrive.

Then, simply put all the scripts in a directory, e.g. `scripts` and the `GSMArena_dataset_2020.csv` in \scripts\dataset.
The `\auxiliary` subdir should also be in there.
The scripts feature_selection, LR, DecisionTree, etc. should run without fail if these steps are met.

#### Options
You may wish to specify the interpolation or imputing with options 'A' & 'B' in clean_data(). The default option is 'B',
which does linear interpolation for columns with fewer than 5000 null values. One may also uncomment cubic or derivative
interpolation instead of linear. 

In feature_selection.py, one may specify 'F' or 'P' for full or partial output of feature expressiveness.

#### Preliminary Feature Selection, 

The modules are: [feature_selection](ml_algorithms/feature_selection.py), [cluster_analysis](ml_algorithms/cluster_analysis.py).

In feature_selection.py, a random forest classifier from sklearn is used as a gauge of the expressiveness of each feature
for random samples & random features. From the results, it appears that features such as the devices 'oem', date of release, dimensions, sensors, screen-body ratio, 
clock speed.

Features that didn't seem as necessary included the device's 'launch status', communication protocols, cpu core count, ROM size, the resolution & video quality of the selfie camera.

#### Auxiliary & Data cleaning

The subdirectory [auxiliary](ml_algorithms/auxiliary) contains python scripts to clean, pipe & transform data into the 
right form. The module [data_clean2.py](ml_algorithms/auxiliary/data_clean2.py) is the main data cleaning script for
getting raw data read from GSMArena_dataset_2020.csv into numerically encoded forms to be immediately input into a 
machine learning algorithm. The module [data_interpolate.py](ml_algorithms/auxiliary/data_interpolate.py) further cleans
up the data and interpolates or imputes missing values.

#### Performance of baseline algorithms

As shown in the results of the Random Forest classifier by sklearn, we have an accuracy of 80-90%. The relationship seems to be either quite complex or perhaps something was missed. The question of whether the data is in the right form is also raised. The much poorer performance of the Multiple Layer Perceptron (<50%) accuracy is also concerning.

The question now is: What really is the relationship between a mobile device's technical specifications and its price? Is it linear, exponential or perhaps much more complex? 

The next sections cover our investigation into a variety of machine learning modules that try and model this relationship as well as possible.

### Experimentation with different machine learning algorithms

#### Linear Regression

Refer to [LR.py](ml_algorithms/LR.py)

It made sense that the one of the first machine learning concept to apply would be a regressor to test if the features
were linearly related to the price. If true, then it shows that the hypothesis would not be too complicated to model accurately.

LR.py contains analysis of batch-trained linear regression algorithms. It also has a custom class for LR batch & gradient descent.
The latter (grad. descent) cannot be run fully & analysis is mostly on batch-training.

#### Logistic Regression

Refer to [LogisticRegression.py](ml_algorithms/LogisticRegression.py)

This contains a logistic regression classifier with gridsearch & assorted plots (included in OneDrive Repo).

#### Decision Trees

Refer to [DecisionTree.py](ml_algorithms/DecisionTree.py)

DecisionTree.py contains a simple decision tree learner on default settings with gridsearch & randomized search
for performance tuning.

#### Support Vector Machines

Refer to [SVM.py](ml_algorithms/SVM.py)

SVM.py contains SVM classifiers (pipelines) and a SV regressor that tries to predict numeric pricing.
An SVM class with half complete SvmMods (individual svm models).

#### k-Nearest Neighbors & Naive-Bayes

Refer to [knn_Bayes.py](ml_algorithms/knn_Bayes.py)

This contains scripts for simple kNN with k=1..11 & Naive-Bayes performance measures. Most other algorithms should be 
able to beat these baselines.
