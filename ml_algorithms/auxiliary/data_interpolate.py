"""
Impute or Interpolate missing data according to categorical & numerical features.
Removes outliers according to m*IQR -> (m=1.5 default).
"""
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import numpy as np
import pandas as pd


# Features

# straight features are straightforward to extract with \d+
# NOTE 1: the current 'straight' feature is battery which seems to be mostly im mAH
# NOTE 2: should still check if its actually 'mAH' and remove or convert if not
straight_features = ["battery"]

all_features = ["oem", "launch_announced", "launch_status", "body_dimensions", "display_size", "comms_wlan", "comms_usb",
                "features_sensors", "platform_os", "platform_cpu", "platform_gpu", "memory_internal",
                "main_camera_single", "main_camera_video", "misc_price",
                "selfie_camera_video",
                "selfie_camera_single", "battery"]

final_features = ["oem", "launch_announced", "body_dimensions", "screen_size", "scn_bdy_ratio",
                  "features_sensors", "clock_speed", "platform_gpu", "ram", "rom",
                  "main_camera_single", "main_camera_video", "misc_price",
                  "selfie_camera_video",
                  "selfie_camera_single", "battery"]

cols_to_drop = ['launch_status',
                'comms_wlan', 'comms_usb', 'platform_os', 'core_count']

numeric_features = ["body_dimensions", "screen_size", "scn_bdy_ratio", "clock_speed", "memory_internal",
                    "main_camera_single", "main_camera_video", "misc_price",
                    "selfie_camera_video",
                    "selfie_camera_single", "battery"]
# TODO: there are some categorical features like 'main_camera_features' & etc. These features should also be included.

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
    df_impute = pd.DataFrame(s_imp.fit_transform(df_ret))
    df_impute.columns = df_ret.columns
    df_impute.index = df_ret.index

    # print("Dimensions of imputed df", df_impute.shape[0], df_impute.shape[1])
    # df_impute.to_csv('imputed_df.csv')
    # print("DF has been output to imputed_df.csv")

    # TEMPORARY: drop cols
    df_ret.dropna(inplace=True)

    # Reindex the data
    df_ret.reset_index(inplace=True)

    return df_ret
