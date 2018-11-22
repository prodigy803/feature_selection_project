# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df):
    rfe = RandomForestClassifier()
    X,y = df.iloc[:,:-1],df.iloc[:,-1]

    selector = RFE(rfe, step=1)
    selector = selector.fit(X, y)

    arr_columns = np.array(df.columns)
    arr = np.select([selector.get_support()],[arr_columns[:-1]] )
    return arr[arr!=0].tolist()










