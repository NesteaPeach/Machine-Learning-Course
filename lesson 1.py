# Libraries that always needed:
# numpy for array management, matplotlib.pyplot for visualization , pandas for data sorting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Libraries for handling missing data
from sklearn.impute import SimpleImputer
# One Hot Encoding - handling string columns and categories
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the csv data into pandas data frame
dataset = pd.read_csv('data/Data.csv')

# how many missing data we have in each column
missing_data = dataset.isnull().sum()

# X is for features and Y is for dependent variable
# the features are all the columns except the last one, the last one is the vector
X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values

# Handle missing data by mean (average):
# initialize the sklearn object
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# fit() - can handle only numerical columns, calculates the mean for the missing data
imputer.fit(X[:, 1:3])

# transform() - returns the columns that were requested without the missing data
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding Categorical Data:
# Encoding the independent variable :
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Encoding the dependent variable
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
