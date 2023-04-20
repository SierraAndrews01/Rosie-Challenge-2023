import pandas as pd
import patsy
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers

xls = pd.ExcelFile('API_ProjectAll_SI.xlsx')
staticDenDataframe = pd.read_excel(xls, 'staticDen')
staticViscDataframe = pd.read_excel(xls, 'staticVisc')

selectStaticDen = staticDenDataframe[['Density', 'T', 'ALogP', 'ALogP2', 'AMR']].copy()
selectStaticVisc = staticViscDataframe[['cP', 'T', 'ALogP', 'ALogP2', 'AMR']].copy()

# set index to be temperature
selectStaticVisc.set_index(['T', 'ALogP', 'ALogP2', 'AMR'], inplace=True)
print(len(selectStaticVisc))
selectStaticDen.set_index(['T', 'ALogP', 'ALogP2', 'AMR'], inplace=True)
print(len(staticDenDataframe))

# grouping relevant data together
finalDataDF = selectStaticVisc.join(selectStaticDen)
#finalDataDF.dropna(0, inplace=True)
# density = selectStaticDen['Density'].to_list()
# finalDataDF['Density'] = pd.Series(density)

print(finalDataDF)

# Helper functions
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
def format_output(data):
    y1 = data.pop('Density')
    y1 = np.array(y1)
    y2 = data.pop('cP')
    y2 = np.array(y2)
    return y1, y2

# splitting data
train, test = train_test_split(finalDataDF, test_size=0.2, random_state=1)
train, val = train_test_split(train, test_size=0.2, random_state=1)

train_stats = train.describe()
print(train_stats)
train_stats.pop('Density')
train_stats.pop('cP')
train_stats = train_stats.transpose()
train_Y = format_output(train)
test_Y = format_output(test)
val_Y = format_output(val)

# Normalize the training and test data
norm_train_X = np.array(norm(train))
norm_test_X = np.array(norm(test))
norm_val_X = np.array(norm(val))

#Analysis
#finalDataDF = finalDataDF.reset_index()
#r2 = r2_score(finalDataDF['Density'], finalDataDF['cP'])
#print(r2)

##rmse = mean_squared_error(test_Y, train_Y, squared = False)
##print(rmse)


#from sklearn.inspection import plot_partial_dependence
# PD Plots
#plot_partial_dependence(model, X, [feature_name])





