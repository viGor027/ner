import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv('../data/ner_datasetreference.csv', encoding='latin1')
print(data.dtypes)
enc = OrdinalEncoder()
tag_col = enc.fit_transform(data['Tag'].to_numpy().reshape(-1, 1))
data['Tag'] = tag_col
print(data.isna().sum())
print(len(data))
data.dropna(inplace=True)
print(len(data))
print(data.isna().sum())
