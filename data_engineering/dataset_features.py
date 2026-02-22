import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np  

df1 = pd.read_csv('datasets/Data/features_30_sec.csv')
df2 = pd.read_csv('datasets/Data/features_3_sec.csv')
print("\nDf1 Statistics: ")
print(df1.head())
print(df1.info())
print(df1.describe())
print(df1.isnull().sum())

print("\nDf2 Statistics: ")
print(df2.head())
print(df2.info())
print(df2.describe())
print(df2.isnull().sum())