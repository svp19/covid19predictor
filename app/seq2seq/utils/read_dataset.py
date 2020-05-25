import pandas as pd
import numpy as np


def read_dataset():
    # Read Dataset
    df = pd.read_csv('../datasets/modified/features_v2.csv', index_col=0)
    df = df.dropna()

    ## Remove Districts with less than 7 days data
    district_window_sizes = df.groupby(['district']).size()
    remove_districts = district_window_sizes[district_window_sizes < 7].index
    df = df[~df['district'].isin(remove_districts)]
    districts = df['district'].unique()

    return df, districts
