import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from auxiliary.data_clean2 import clean_data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from feature_selection import y_classify_five, y_classify
# Load up dataset 1: gsmarena
data = pd.read_csv('dataset/GSMArena_dataset_2020.csv', index_col=0)

data_features = data[["oem", "launch_announced", "launch_status", "body_dimensions", "display_size", "comms_wlan", "comms_usb",
                "features_sensors", "platform_os", "platform_cpu", "platform_gpu", "memory_internal",
                "main_camera_single", "main_camera_video", "misc_price",
                "selfie_camera_video",
                "selfie_camera_single", "battery"]]

# Clean up the data into a trainable form.
df = clean_data(data_features)

df.dropna(inplace=True)
df.reset_index(drop=True)

y = df["misc_price"]
Xy = df.drop(["key_index"], axis=1)
X = df.drop(["key_index", "misc_price"], axis=1)


# convert to categorical data
lab_enc = preprocessing.LabelEncoder()


features = ["oem", "launch_announced",  "body_dimensions", "features_sensors",  "platform_gpu", "main_camera_single", \
            "main_camera_video", "selfie_camera_video","selfie_camera_single", "battery", "clock_speed", \
            "screen_size", "scn_bdy_ratio", "rom", "ram", "misc_price"]



sns.set_style("whitegrid")
sns.boxplot(x=y) #Box plot
plt.show()

# for i in range(len(features)-1):
#     sns.regplot(x=features[i],y=y,data=X)
#     plt.show()


y = y.apply(y_classify)

#Cite: COMP9417 Tutorial1 Lab2_Linear_Regression.ipynb
# Compute the correlation matrix
corr = Xy.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()






# Split data into train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

'''
TO DO: To reduce dimensions, select one to three attributes that have the most impact. Use correlation heatmap to find out
'''


param_grid = {'n_neighbors': np.arange(1,11), 'weights': ['uniform', 'distance'],'algorithm' : ['auto','ball_tree', 'kd_tree','brute']}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, return_train_score=True)
grid_knn.fit(X_train, y_train)

estimator = grid_knn.best_estimator_
y_pred = grid_knn.predict(X_test)
result = accuracy_score(y_test, y_pred)
print('the best result is: ',result,' and the param is:\n',best_params_)

# for i in range(1, 11):
#     clf = KNeighborsClassifier(n_neighbors=i, weights='distance')
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print('\n- - - k = ',i)
#     print(classification_report(y_test, y_pred))
#     print('precise accuracy = ',accuracy_score(y_pred, y_test))

#plot data 
