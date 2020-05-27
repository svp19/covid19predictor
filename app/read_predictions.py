import pandas as pd


def read_predictions(district):
    district = district.replace(' ', '')
    district = district.lower()
    df = pd.read_csv(f'app/predictions/{district}.csv')
    return df
