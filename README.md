# Predicting the value of Mobile devices from their specifications

## Preliminary Feature Selection

In feature_selection.py, a random forest classifier from sklearn is used as a gauge of the expressiveness of each feature
for random samples & random features. From the results, it appears that features such as the devices 'oem', date of release, dimensions, sensors, screen-body ratio, 
clock speed.

Features that didn't seem as necessary included the device's 'launch status', communication protocols, cpu core count, the resolution of the selfie camera.

#### Performance of sklearn's algorithms

As shown in the results of the Random Forest classifier by sklearn, we have an accuracy of 80%. The relationship seems to be either quite complex or perhaps something was missed. The question of whether the data is in the right form is also raised. The much poorer performance of the Multiple Layer Perceptron at 10% accuracy is also concerning.

The question now is: What really is the relationship between a mobile device's technical specifications and its price? Is it linear, exponential or perhaps much more complex? 

The next sections cover our investigation into a variety of machine learning algorithms that try and model this relationship as well as possible.

## Linear Regression

It made sense that the one of the first machine learning concept to apply would a regressor that to test if the features were linearly related to the price.

If it was, then it shows that the hypothesis should not be too complicated.
