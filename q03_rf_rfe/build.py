# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df=data):
    rfe = RandomForestClassifier()
    X,y = data.iloc[:,:-1],data.iloc[:,-1]

    selector = RFE(rfe, step=1)
    selector = selector.fit(X, y)

    arr_columns = np.array(data.columns)
    arr = np.select([selector.get_support()],[arr_columns[:-1]] )
    return [arr[arr!=0]]










