import pandas as pd
import patsy
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import sklearn.inspection as skl
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import wrappers
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pdpbox
import traceback

xls = pd.ExcelFile('API_ProjectAll_SI.xlsx')
staticDenDataframe = pd.read_excel(xls, 'staticDen')
staticViscDataframe = pd.read_excel(xls, 'staticVisc')

selectStaticDen = staticDenDataframe[['Density', 'T', 'ALogP', 'ALogP2', 'AMR']].copy()
selectStaticVisc = staticViscDataframe[['cP', 'T', 'ALogP', 'ALogP2', 'AMR']].copy()

# set index to be temperature
selectStaticVisc.set_index(['T', 'ALogP', 'ALogP2', 'AMR'], inplace=True)
selectStaticDen.set_index(['T', 'ALogP', 'ALogP2', 'AMR'], inplace=True)

# df1 = larger dataframe
# df2 = smaller dataframe
def makeSameShape(df1, df2):

    # create two dataframes with different shapes
    # df1 = pd.DataFrame(np.random.randn(5, 3), columns=['A', 'B', 'C'])
    # df2 = pd.DataFrame(np.random.randn(3, 2), columns=['D', 'E'])

    # print the original shapes
    print("Shape of df1:", df1.shape)
    print("Larger Dataset Head")
    print(df1.head())
    print("Shape of df2:", df2.shape)
    print("Smaller Dataset Head")
    print(df2.head())

    # reindex df2 to match the shape of df1
    df1 = df1.reindex(index=df2.index, columns=df2.columns)

    # print the new shapes
    print("Shape of df1:", df1.shape)
    print("Shape of df2:", df2.shape)

    return df1, df2

def get_train_stats(x_train):
    train_stats = x_train.describe()
    if x_train.equals(X1_train):
        train_stats.pop('cP')
    elif x_train.equals(X2_train):
        train_stats.pop('Density')
    train_stats = train_stats.transpose()
    return train_stats

# find the dataframe with more samples
if len(selectStaticVisc) > len(selectStaticDen):
    larger_df = selectStaticVisc
    smaller_df = selectStaticDen
    print("Static density is smaller")
    print("Size of static den: ", len(selectStaticDen))

    print("First time: ", larger_df.index.is_unique)
    print("First time: ", smaller_df.index.is_unique)

    larger_df.reset_index(drop=True, inplace=True)
    smaller_df.reset_index(drop=True, inplace=True)

    print("Second time: ", larger_df.index.is_unique)
    print("Second time: ", smaller_df.index.is_unique)

    larger_df, smaller_df = makeSameShape(larger_df, smaller_df)

elif len(selectStaticDen) > len(selectStaticVisc):
    larger_df = selectStaticDen
    smaller_df = selectStaticVisc
    print("Static viscosity is smaller")
    print("Size of static visc: ", len(selectStaticVisc))
    print("Size of static den: ", len(selectStaticDen))

    print("First time: ", larger_df.index.is_unique)
    print("First time: ", smaller_df.index.is_unique)

    larger_df.reset_index(drop=True, inplace=True)
    smaller_df.reset_index(drop=True, inplace=True)

    print("Second time: ", larger_df.index.is_unique)
    print("Second time: ", smaller_df.index.is_unique)

    print("ldfs1: ", larger_df.shape)
    print("sdfs1: ", smaller_df.shape)

    larger_df, smaller_df = makeSameShape(larger_df, smaller_df)

    print("ldfs2: ", larger_df.shape)
    print("sdfs2: ", smaller_df.shape)

else:
    print("Lengths are equal")
    print("StaticDen Len: ", len(selectStaticDen))
    print("StaticVisc Len: ", len(selectStaticVisc))

    makeSameShape(selectStaticDen, selectStaticVisc)

# df1 = smaller dataframe
# df2 = larger dataframe


# randomly sample rows from the larger dataframe to match the smaller dataframe
#num_samples = len(smaller_df)
#print("Larger dataframe before modified: ", larger_df.shape())

#larger_df = larger_df.sample(n=num_samples)

# print("Length of smaller df: ", len(smaller_df))
# print("length of larger df: ", len(larger_df))
#
# larger_df.head()
# selectStaticDen.head()


# concatenate the two dataframes into a single dataframe
# combined_df = pd.concat([smaller_df, larger_df], axis=0)
# print("Length of combined df: ", len(combined_df))
#
#
# # grouping relevant data together
# finalDataDF = selectStaticVisc.join(selectStaticDen)
# finalDataDF = finalDataDF.dropna()
# finalDataDF = finalDataDF.reset_index()
# print("Length of final dataframe: ", len(finalDataDF))
#print(finalDataDF)
#print(finalDataDF.isna().any().any())

# Helper functions
def norm(x):
    train_stats = get_train_stats(x_train=x)
    return (x - train_stats['mean']) / train_stats['std']
    #return 0
def format_output(data):
    y1 = data.pop('Density')
    y1 = np.array(y1)
    y2 = data.pop('cP')
    y2 = np.array(y2)
    return y1, y2

# splitting data
#train, test = train_test_split(finalDataDF, test_size=0.2, random_state=1)
#train, val = train_test_split(train, test_size=0.2, random_state=1)


# Creates a numpy tuple that is the same size as the smaller dataframe
n_samples = len(smaller_df)
y = np.full((n_samples, 1), 2)
print("Y size: ", y.shape)

# Assuming you have two input arrays X1 and X2, and one output array y
# larger_df = X1
# smaller_df = X2
# test_size is 20% of data
# random_state ensures that the same random shuffling is used each time (this is useful for
#   reproducibility (We chose a random number but might change this number to 1)
try:
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(larger_df,
                                                                         smaller_df,
                                                                         y,
                                                                         test_size=0.2,
                                                                         random_state=42)
    print("SUCCESS!!!")
    print("X1 Train")
    print(X1_train)
    print('---------------------------------------------------------')
    print("X2 Train")
    print(X2_train)
except Exception as e:
    print("Size of Static Den: ", len(selectStaticDen))
    print("Size of Static Visc: ", len(selectStaticVisc))
    print("Size of y: ", len(y))
    traceback.print_exc()


# Use X1_train and X2_train as the input to the neural network during training,
# and X1_test and X2_test as the input to the neural network during testing


# train_stats = train.describe()
# print(train_stats)
# train_stats.pop('Density')
# train_stats.pop('cP')
# train_stats = train_stats.transpose()
# train_Y = format_output(train)
# test_Y = format_output(test)
# val_Y = format_output(val)


# Normalize the training and test data
# Input 1
norm_train_X1 = np.array(norm(X1_train))
norm_test_X1 = np.array(norm(X1_test))
#Input 2
norm_train_X2 = np.array(norm(X2_train))
norm_test_X2 = np.array(norm(X2_test))
#Output
norm_train_Y = np.array(norm(y_train))
norm_test_Y = np.array(norm(y_test))


#norm_val_X = np.array(norm(val))





#print("Number of samples in input data: ", norm_train_X.shape[0])
#len(data)
#print("Number of samples in target data: ", train_Y.shape[0])

#Analysis
#finalDataDF = finalDataDF.reset_index()
# r2 = r2_score(finalDataDF['Density'], finalDataDF['cP'])
# print(r2)
#
# rmse = mean_squared_error(test_Y, train_Y, squared = False)
# print(rmse)

# print shape of input data
print("Train X1 Shape: ", norm_train_X1.shape)
print("Test X1 Shape: ", norm_test_X1.shape)
print("Train X2 Shape: ", norm_train_X2.shape)
print("Test X2 Shape: ", norm_test_X2.shape)
print("Train Y Shape: ", norm_train_Y.shape)
print("Test Y Shape: ", norm_test_Y.shape)

#print(norm_val_X.shape)

# NNN
# Define the model architecture
skl.partial_dependence(estimator=model, X=norm_test_X, features=[0, 1, 2, 3])