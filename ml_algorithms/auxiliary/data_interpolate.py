"""
Impute or Interpolate missing data according to categorical & numerical features.
Removes outliers according to m*IQR -> (m=1.5 default).
TODO: Add other functionality where fit.
"""
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import numpy as np
import pandas as pd


# Features
straight_features = ["memory_internal",
                     "main_camera_single", "main_camera_video",
                     "selfie_camera_video",
                     "selfie_camera_single", "battery"]

all_features = ["oem", "launch_announced", "launch_status", "body_dimensions", "display_size", "comms_wlan", "comms_usb",
                "features_sensors", "platform_os", "platform_cpu", "platform_gpu", "memory_internal",
                "main_camera_single", "main_camera_video", "misc_price",
                "selfie_camera_video",
                "selfie_camera_single", "battery"]

final_features = ["oem", "launch_announced", "launch_status", "body_dimensions", "screen_size", "scn_bdy_ratio", "comms_wlan", "comms_usb",
                  "features_sensors", "platform_os", "core_count", "clock_speed", "platform_gpu", "memory_internal",
                  "main_camera_single", "main_camera_video", "misc_price",
                  "selfie_camera_video",
                  "selfie_camera_single", "battery"]

numeric_features = ["body_dimensions", "screen_size", "scn_bdy_ratio", "clock_speed", "memory_internal",
                    "main_camera_single", "main_camera_video", "misc_price",
                    "selfie_camera_video",
                    "selfie_camera_single", "battery"]


def rem_outliers(df):
    for feature in numeric_features:
        series_ = df[feature]
        # Calc IQR for the column
        Q1 = np.quantile(series_, .75)
        Q3 = np.quantile(series_, .25)
        out_factor = 1.5*(Q3-Q1)
        out_cond = lambda x: x and (x >= Q1 - out_factor or x <= Q3 - out_factor)

        # Check if each value is > 1.5 * IQR
        # NOTE: does not work if a lot of examples were not properly recorded/extracted
        df[feature] = series_.apply(lambda x: x if out_cond(x) else np.nan)
        
    # Return outlier free data, at the cost of potentially many missing examples.
    print(df.shape[0], df.shape[1])
    return df


# TODO: Since we are basically trying to interpolate over 90% of the missing data points
# It may be better to just use the ~700-800 examples available after outlier removal.
def fill_gaps(df):
    # NOTE: Can also use some interpolation (linear, cubic) instead.
    i_imp = IterativeImputer(max_iter=20, random_state=6)
    s_imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Infer the object types -> ensuring numeric encoding.
    # NOTE: categorical/numeric split pipeline?
    df = df.fillna(value=np.nan)
    df_ret = df.infer_objects()
    for feature in final_features:
        df_ret[feature] = pd.to_numeric(df_ret[feature], downcast='float')

    # Remove outliers for each column, if they are 1.5X IQR for the column.
    # df_ret = rem_outliers(df_ret)

    # Impute missing data, i.e. NaN.
    # df_ret[df_ret.columns] = s_imp.fit_transform(df_ret[df_ret.columns])

    # TODO: Apply smoothing function (exponential, gaussian).
    

    # TEMPORARY: drop cols
    df_ret.dropna(inplace=True)

    # Reindex the data
    df_ret.reset_index(inplace=True)

    return df_ret
