import pandas as pd


def import_winedata():
    df = pd.read_csv('../wine_quality_dataset/winequality-red.csv', sep=';')
    if df.isnull().values.any():
        print("Missing Values Found")  # check for missing values
    else:
        print("No Missing Values Found")

    features = df.drop(columns='quality')
    labels = df['quality']
    return features, labels
