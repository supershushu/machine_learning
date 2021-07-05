#!/usr/bin/env python

import pandas as pd
import os
from model import Model


def load_data():
    path = '/Users/jinma/Data/NO2_MaJin'
    files = os.listdir(path)
    print(files)
    file1 = 'A_MIXHI_HR_HKO_VT_223200_1141700-2002.csv'
    df = pd.read_csv(os.path.join(path, file1), skiprows=3)

    print(df.info())


def run_model():
    model = Model()
    model()


if __name__ == '__main__':
    run_model()
