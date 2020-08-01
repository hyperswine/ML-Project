"""
A revised version of data clean. Now the script itself is cleaner & less problematic to maintain.
This script should drop examples (rows) with outliers, i.e. 1.5X IQR.
It should then impute all the columns such we maintain most of the examples.
"""
import re
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import LabelEncoder

from .data_interpolate import *


def extract_straight(df):
    for feature in straight_features:
        s = df[feature]
        df[feature] = s.str.extract(r"(\d+)")

    return df


def extract_f(df):
    for feature in f_map:
        f = f_map.get(feature)
        df[feature] = df[feature].apply(f)

    return df


##########################
# FUNCTION DEFINTIONS
##########################

def launch_announced(string):
    year = re.search(r"\d{4}", str(string))
    return year.group(0) if year else None


def available_discontinued(string):
    return str(1) if "Available" in string else str(0)


def usb_type(string):
    string = str(string)
    if ("31" in string or "30" in string) and "Type-C" in string:
        return str(3)
    if "Type-C" in string:
        return str(2)
    if "proprietary" in string.lower():
        return str(1)

    return str(0)


def squared_dimensions(string):
    dimensions_m = string.split('m')
    dimensions = [t for t in dimensions_m[0].split()
                  if t.lstrip('+-').replace('.', '', 1).isdigit()]

    try:
        return str(reduce(lambda r, d: float(r) * float(d), dimensions))
    except:
        # print(f"could not retreive dimensions^2, for {string}")
        return None


def wlan(string):
    string = string.lower()

    if '6' in string and 'wifi' in string:
        return str(4)
    if 'ac' in string and 'wifi' in string:
        return str(3)
    if 'direct' in string or 'hotspot' in string:
        return str(2)
    if 'wifi' in string:
        return str(1)

    return str(0)


def sensor(string):
    sensors = ['accelerometer', 'proximity',
               'compass', 'gyro', 'fingerprint', 'barometer']

    return len([ sensor for sensor in sensors if (sensor in string.lower()) ])


def cam_vid(string):
    string = str(string)

    p_string = re.search(r"\d+p", string)
    k_string = ""

    if not p_string:
        k_string = re.search(r"\d+", string.lower())
        if not k_string:
            return string
        if '2' in k_string.group(0):
            return '1080p'
        if '4' in k_string.group(0):
            return '2160p'

    return string


def os(string):
    string = str(string)
    if "Android" in string:
        return str(3)
    if "iOS" in string:
        return str(2)
    if "Microsoft" in string:
        return str(1)

    return str(0)


def gpu_platform(string):
    string = str(string)

    cpu_arch = ["powervr", "mali", "adreno"]
    for cpu in cpu_arch:
        if cpu in string.lower():
            return str(cpu_arch.index(cpu) + 1)

    return str(0)


def extract_screen_in(df):
    """
    Add two new columns, screen_size & scn_bdy_ratio, remove 'display_size'.
    If not found, then just add 'None'.
    """
    # Regex for screen size in inches
    df['screen_size'] = df['display_size'].apply(
        lambda x: re.search(r'^.*(?=( inches))', str(x).lower()) )

    # Regex for screen-body ratio
    df['scn_bdy_ratio'] = df['display_size'].apply(
        lambda x: re.search(r'\d{1,2}.\d(?=%)', str(x).lower()) )

    # Apply results, NOTE: pandas doesn't like it when we're applying to multiple series.
    results1 = df['scn_bdy_ratio'].apply(lambda y: y.group(0) if y else None)
    results2 = df['screen_size'].apply(lambda y: y.group(0) if y else None)
    df['scn_bdy_ratio'] = results1
    df['screen_size'] = results2

    print(df['scn_bdy_ratio'], df['screen_size'])

    return df.drop(['display_size'], axis=1)


def core_count(string):
    cores = ["Dual", "Quad", "Octa"]
    count = 0
    for core in cores:
        if core in str(string):
            count = cores.index(core) + 1
            break

    return str(count)


def extract_cpu(df):
    """
    Split 'platform_cpu' to 'core_count' and 'clock_speed', drop 'platform_cpu'
    """
    # Get core count; i.e. first string.
    df['core_count'] = df['platform_cpu'].apply(core_count)

    # Split by spaces & make everything lower case
    s_l = df['platform_cpu'].apply(lambda x: str(x).lower())

    # Get clock speed, i.e. '\d g|mhz'
    s_l = s_l.apply(get_clk_speed)

    # If in ghz, multiply by 1000 & remove decimal place & convert all to float
    s_l = s_l.apply(ghz_to_mhz)
    df['clock_speed'] = s_l.apply(lambda x: float(x.split()[0]) if x else None)

    return df.drop(['platform_cpu'], axis=1)


# Check for mhz/ghz & retreive with regex
def get_clk_speed(string):
    t = None
    if "mhz" in string:
        t = re.search(r'\d+ mhz', string)
    elif "ghz" in string:
        t = re.search(r'(\d+\.\d+) ghz', string)

    return t.group(0) if t else None


# Convert strings that have 'ghz' to '1000*x mhx' where x is in ghz.
def ghz_to_mhz(string):
    if not string or 'ghz' not in string: return string

    temp = float(string.split()[0])

    return str(temp*1000) + ' mhz'
    

# Function Map
f_map = {"launch_announced": launch_announced, "launch_status": available_discontinued,
         "body_dimensions": squared_dimensions, "comms_wlan": wlan,
         "comms_usb": usb_type,
         "features_sensors": sensor, "platform_os": os,
         "platform_gpu": gpu_platform}


def clean_data(df):

    df_ret = pd.DataFrame()

    # Get all numeric values obtained by the first regexed number(s).
    df = extract_straight(df)

    # Get all other numeric & categorical features specifically.
    df = extract_f(df)
    df = extract_cpu(df)
    df = extract_screen_in(df)

    # Encode 'OEM' with label-encoder after lower().
    oem = df.oem.apply(lambda string: ''.join(c for c in string if c.isalnum()).lower())
    oem = oem.apply(lambda x: str(x))
    enc = LabelEncoder()
    df['oem'] = enc.fit_transform(oem)
    df['oem'] = df.oem.apply(pd.to_numeric)

    # Impute missing data & remove outliers
    df_ret = fill_gaps(df)

    # Return final dataframe containing cleaned & filled data
    return df_ret


if __name__ == '__main__':

    # Open Dataset
    data = pd.read_csv('C:/Users/capta/Desktop/Project-Report/ml_algorithms/dataset/GSMArena_dataset_2020.csv', index_col=0)

    # Extract relevant features (for now)
    data_features = data[all_features]

    # Clean data
    dfX = clean_data(data_features)

    # Output insights on the final data.
    print(dfX.info())
    print(dfX.head())
