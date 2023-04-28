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
#from tensorflow.keras import layers
from keras import layers
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pdpbox

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
finalDataDF = finalDataDF.dropna()
finalDataDF = finalDataDF.reset_index()
print(len(finalDataDF))
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

# print shape of input data
print(norm_train_X.shape)
print(norm_val_X.shape)
print(norm_test_X.shape)

# NNN
# Define the model architecture
inputs = keras.Input(shape=(4,))
x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(16, activation="relu")(x)

output1 = layers.Dense(1, name="Density")(x)
output2 = layers.Dense(1, name="cP")(x)

model = keras.Model(inputs=inputs, outputs=[output1, output2])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss={
        "Density": "mean_squared_error",
        "cP": "mean_squared_error"
    },
    metrics={
        "Density": "mean_absolute_error",
        "cP": "mean_absolute_error"
    }
)

# Train the model
history = model.fit(
    x=norm_train_X,  # update the input parameter name to "x"
    y=train_Y,  # update the output parameter name to "y"
    batch_size=32,
    epochs=100,
    validation_data=(norm_val_X, val_Y)
)

# Evaluate the model on the test set
model.evaluate(norm_test_X, test_Y)


# Create PDP plots
# create the PDP plots for input variables 'T', 'ALogP', 'ALogP2', and 'AMR' with respect to output variable 'Density'
skl.partial_dependence(estimator=model, X=norm_test_X, features=[0, 1, 2, 3])

# create the PDP plots for input variables 'T', 'ALogP', 'ALogP2', and 'AMR' with respect to output variable 'cP'
skl.partial_dependence(estimator=model, X=norm_test_X, features=[0, 1, 2, 3])