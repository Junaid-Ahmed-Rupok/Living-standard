import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def preprocessing(read_csv):
    df = pd.read_csv(read_csv)  # converting the csv file into a dataframe

    for cols in df.columns:  # dealing with unexpected features
        if df[cols].isna().sum() > df.shape[0] * 0.8:
            df = df.drop(columns=[cols])

    if df.isna().sum().sum() > 0:
        df = df.apply(
            lambda cols: (
                cols.fillna(cols.mean())
                if np.issubdtype(cols.dtype, np.number)
                else cols.fillna(cols.mode()[0])
            )
        )  # dealing with null values

    if df.duplicated().sum():
        df = df.drop_duplicates()  # dealing with duplicates

    encoder = OrdinalEncoder()
    df_cats = df.select_dtypes(
        include=["object"]
    ).columns.tolist()  # categorical features of the dataframe
    df[df_cats] = encoder.fit_transform(
        df[df_cats]
    )  # encoding the categorical features
    joblib.dump(encoder, "encoder.pkl")  # dumping the encoder

    return df


cleaned_df = preprocessing(r"csv_files/Living_Standard.csv")
joblib.dump(cleaned_df, "df.pkl")  # dumping the dataframe
