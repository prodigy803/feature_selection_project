# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df,k=20):
    X = df.iloc[:,:-1]
    y= df.iloc[:,-1]
    selection = SelectPercentile(f_regression, percentile=k)
    x_features = selection.fit_transform(X, y)
    columns = np.asarray(X.columns.values)
    support = np.asarray(selection.get_support())
    columns_with_support = columns[support]
    return list(columns_with_support)







