import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Read in data
dtfm=pd.read_excel('initial_data.xlsx', sheet_name='BD_Research_Fapesp_final', header=1,usecols=[4,5,32,33,34])

print(dtfm.head())
