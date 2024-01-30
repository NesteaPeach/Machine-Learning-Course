# Libraries that always needed:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the csv data into pandas data frame
dataset = pd.read_csv('data/Data.csv')


# X is for features and Y is for dependent variable
# the features are all the columns except the last one, the last one is the vector
x = dataset.iloc[:, : -1]
y = dataset.iloc[:, -1]
