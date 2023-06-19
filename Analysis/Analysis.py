# Import necessary libraries
import numpy as np
import pandas as pd
from keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pdpbox import pdp
from sklearn.base import BaseEstimator, RegressorMixin
from lime.lime_tabular import LimeTabularExplainer
import lime
import lime.lime_tabular
import shap
import h5py


# Function to plot the data
def plotdata(datax, datay, labelx, labely, limitx, limity, figurename):
    font = {'family': 'Arial', 'weight': 'bold', 'size': 24}
    tickfont = {'family': 'Arial', 'weight': 'bold', 'size': 20}
    tick_width = 2
    tick_length = 10
    plot_box_thickness = 2

    # Plot darkness vs frame
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(datax, datay)
    ax.set_xlabel(labelx, fontdict=font)
    ax.set_ylabel(labely, fontdict=font)
    ax.tick_params(axis='both', which='major', labelsize=tickfont['size'],
                   length=tick_length, width=tick_width, direction='in')

    # Set plot box thickness
    for spine in ax.spines.values():
        spine.set_linewidth(plot_box_thickness)

    if limitx:
        ax.set_xlim(limitx)
    if limity:
        ax.set_ylim(limity)

    plt.show()
    #image_path = f'C:\\Users\\andrewss\\PycharmProjects\\Rosie-Challenge-2023\\Data PNGs\\{figurename}.png'
    #fig.savefig(fname=image_path, dpi=300, bbox_inches='tight')


# Function to calculate VIF for a dataset
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["features"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif


# Function to calculate RMSE and R-squared
def evaluate_model(model, X, Y):
    Y_pred = model.predict(X)
    density_rmse = np.sqrt(mean_squared_error(Y[0], Y_pred[0]))
    viscosity_rmse = np.sqrt(mean_squared_error(Y[1], Y_pred[1]))
    density_r2 = r2_score(Y[0], Y_pred[0])
    viscosity_r2 = r2_score(Y[1], Y_pred[1])
    return density_rmse, viscosity_rmse, density_r2, viscosity_r2


# Helper functions
def norm(x):
    # Normalize the data using the mean and standard deviation of the training set
    return (x - train_stats['mean']) / train_stats['std']


def format_output(data):
    # Extract and format the target variables (density and viscosity)
    y1 = data.pop('Density')
    y1 = np.array(y1)
    y2 = data.pop('cP')
    y2 = np.array(y2)
    return [y1, y2]

# Create file to write results to:
file = open("Data PNGs/Rosie Data Analysis", "w")


# Load the Excel file
xls = pd.ExcelFile('C:\\Users\\andrewss\\PycharmProjects\\Rosie-Challenge-2023\\Analysis\\API_ProjectAll_SI.xlsx')
staticDenDataframe = pd.read_excel(xls, 'staticDen')
staticViscDataframe = pd.read_excel(xls, 'staticVisc')

# Select the relevant columns # 𝑇, 𝑝𝑖𝑃𝐶3, 𝑅𝑜𝑡𝐵𝑡𝐹𝑟𝑎𝑐, 𝑇, 𝐴𝐴𝑇𝑆2𝑒, 𝑇, 𝑅𝑜𝑡𝐵𝐹𝑟𝑎𝑐, 𝑀𝑊, 𝑃𝑒𝑡𝑖𝑡𝑗𝑒𝑎𝑛𝑁𝑢𝑚𝑏𝑒𝑟
selectStaticDen = staticDenDataframe[['Density', 'T', 'ALogP', 'ALogP2', 'AMR']].copy()
selectStaticVisc = staticViscDataframe[['cP', 'T', 'ALogP', 'ALogP2', 'AMR']].copy()

# Set the index to be temperature and other relevant columns
selectStaticVisc.set_index(['T', 'ALogP', 'ALogP2', 'AMR'], inplace=True)
selectStaticDen.set_index(['T', 'ALogP', 'ALogP2', 'AMR'], inplace=True)

# Group the relevant data together
finalDataDF = selectStaticVisc.join(selectStaticDen)
finalDataDF = finalDataDF.dropna()
finalDataDF = finalDataDF.reset_index()

# Split the data into train, validation, and test sets
train, test = train_test_split(finalDataDF, test_size=0.2, random_state=1)
train, val = train_test_split(train, test_size=0.2, random_state=1)

# Calculate summary statistics for the training data
train_stats = train.describe()
train_stats.pop('Density')
train_stats.pop('cP')
train_stats = train_stats.transpose()

# Extract target variables for train, test, and validation sets
train_Y = format_output(train)
test_Y = format_output(test)
val_Y = format_output(val)

# Define feature names
feature_names = train.columns

# Normalize the training and test data
norm_train_X = np.array(norm(train))
norm_test_X = np.array(norm(test))
norm_val_X = np.array(norm(val))

# Define the neural network architecture
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
    loss={"Density": "mean_squared_error",
          "cP": "mean_squared_error"},
    metrics={"Density": "mean_absolute_error",
             "cP": "mean_absolute_error"}
)

# Train the model
history = model.fit(
    x=norm_train_X,
    y=train_Y,
    batch_size=32,
    epochs=100,
    validation_data=(norm_val_X, val_Y)
)

# Save model to JSON file
json_model = model.to_json()
print("Model file successfully created")
#save the model architecture to JSON file
with open('Data PNGs/modelFile.json', 'w') as json_file:
    json_file.write(json_model)
print("Architecture saved")
#saving the weights of the model
model.save_weights('ModelWeights/modelFileWeights.h5py')
print("Model weights saved")

# Evaluate the model on the validation set
loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=norm_val_X, y=val_Y)

"""
# Print the evaluation metrics
print()
print(f'loss: {loss}')
print(f'price_loss: {Y1_loss}')
print(f'ptratio_loss: {Y2_loss}')
print(f'price_rmse: {Y1_rmse}')
print(f'ptratio_rmse: {Y2_rmse}')
"""

# Make predictions on the test set
Y_pred = model.predict(norm_test_X)
density_pred = Y_pred[0]
viscosity_pred = Y_pred[1]

datax = test_Y[0]
datay = Y_pred[0]
labelx = 'Actual Density'
labely = 'Predicted Density'
figurename = 'Density'
mindensity = np.min([np.min(test_Y[0]), np.min(Y_pred[0])])
maxdensity = np.max([np.max(test_Y[0]), np.max(Y_pred[0])])
limitx = [mindensity, maxdensity];
limity = [mindensity, maxdensity];
plotdata(datax, datay, labelx, labely, limitx, limity, figurename)

datax = test_Y[1]
datay = Y_pred[1]
labelx = 'Actual Viscosity'
labely = 'Predicted Viscosity'
figurename = 'Viscosity'
minviscosity = np.min([np.min(test_Y[1]), np.min(Y_pred[1])])
maxviscosity = np.max([np.max(test_Y[1]), np.max(Y_pred[1])])
limitx = [minviscosity, maxviscosity];
limity = [minviscosity, maxviscosity];
plotdata(datax, datay, labelx, labely, limitx, limity, figurename)

# Calculate VIF for the train data set (before normalization)
train_X = train.copy()
vif = calculate_vif(train_X)
print("VIF for train data set:")
print(vif)
file.write('VIF for train data set:\n')
file.write(f'{vif} \n\n')

# Calculate RMSE and R-squared for train, validation, and test data sets
train_density_rmse, train_viscosity_rmse, train_density_r2, train_viscosity_r2 = evaluate_model(model, norm_train_X,
                                                                                                train_Y)
val_density_rmse, val_viscosity_rmse, val_density_r2, val_viscosity_r2 = evaluate_model(model, norm_val_X, val_Y)
test_density_rmse, test_viscosity_rmse, test_density_r2, test_viscosity_r2 = evaluate_model(model, norm_test_X, test_Y)

"""
vif = pd.DataFrame()
    vif["features"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif
"""


# Print the evaluation metrics
print("RMSE and R-squared for train data set:\n")
print(f'Density RMSE: {train_density_rmse}, Viscosity RMSE: {train_viscosity_rmse}')
print(f'Density R-squared: {train_density_r2}, Viscosity R-squared: {train_viscosity_r2}')

print("RMSE and R-squared for validation data set:")
print(f'Density RMSE: {val_density_rmse}, Viscosity RMSE: {val_viscosity_rmse}')
print(f'Density R-squared: {val_density_r2}, Viscosity R-squared: {val_viscosity_r2}')

print("RMSE and R-squared for test data set:")
print(f'Density RMSE: {test_density_rmse}, Viscosity RMSE: {test_viscosity_rmse}')
print(f'Density R-squared: {test_density_r2}, Viscosity R-squared: {test_viscosity_r2}')

##################### Print the above to a file for easier reading ############################
train_data = pd.DataFrame()
train_data['Variable'] = ['RMSE', 'R-Squared']
train_data['Density'] = [train_density_rmse, train_density_r2]
train_data['Viscosity'] = [train_viscosity_rmse, train_viscosity_r2]
train_data.set_index('Variable', inplace=False)
file.write('RMSE and R-squared for training data set:\n'
           f'{train_data} \n\n')

val_data = pd.DataFrame()
val_data['Variable'] = ['RMSE', 'R-Squared']
val_data['Density'] = [val_density_rmse, val_density_r2]
val_data['Viscosity'] = [val_viscosity_rmse, val_viscosity_r2]
file.write('RMSE and R-squared for validation data set:\n'
            f'{val_data} \n\n')

test_data = pd.DataFrame()
test_data['Variable'] = ['RMSE', 'R-Squared']
test_data['Density'] = [test_density_rmse, test_density_r2]
test_data['Viscosity'] = [test_viscosity_rmse, test_viscosity_r2]
file.write('RMSE and R-squared for test data set:\n'
           f'{test_data} \n\n')

# Plot training & validation loss values
epochs = 100
fig, ax = plt.subplots(figsize=(10, 5), sharex=True)
plt.plot(history.history["loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.grid()
plt.savefig("Data PNGs/ModelLoss")
plt.show()

####################################### PARTIAL DEPENDENCY PLOTS ########################################
# Function to create a model for a single output
def create_single_output_model(model, output_index):
    new_model = keras.Model(inputs=model.inputs, outputs=model.outputs[output_index])
    return new_model

# Create separate models for density and viscosity
density_model = create_single_output_model(model, 0)
viscosity_model = create_single_output_model(model, 1)

# Define feature names
feature_names = train.columns

print(train.columns)

# Custom normalization function that returns a Pandas DataFrame
def norm_with_columns(x):
    return pd.DataFrame((x - train_stats['mean']) / train_stats['std'], columns=feature_names)

# Normalize the training data and retain column names
norm_train_X_df = norm_with_columns(train)

# Function to plot PDP for a specific feature

def plot_pdp_density(model, dataset, feature_name):
    pdp_feature = pdp.pdp_isolate(
        model=model,
        dataset=dataset,
        model_features=feature_names,
        feature=feature_name
    )
    pdp.pdp_plot(pdp_feature, feature_name)
    plt.savefig('Data PNGs/density' + feature_name)
    plt.show()


def plot_pdp_viscosity(model, dataset, feature_name):
    pdp_feature = pdp.pdp_isolate(
        model=model,
        dataset=dataset,
        model_features=feature_names,
        feature=feature_name
    )
    pdp.pdp_plot(pdp_feature, feature_name)
    plt.savefig('Data PNGs/viscosity' + feature_name)
    plt.show()


# Plot PDP for each feature in the dataset for density
for feature in feature_names:
    plot_pdp_density(density_model, norm_train_X_df, feature)
    print()

# Plot PDP for each feature in the dataset for viscosity
for feature in feature_names:
    plot_pdp_viscosity(viscosity_model, norm_train_X_df, feature)
    print()

####################################### LIME and SHAPLEY VALUES ########################################

# Define a function to create a LIME explainer for a specific output
def create_lime_explainer(train_X, train_Y, feature_names):
    return lime.lime_tabular.LimeTabularExplainer(
        train_X,
        training_labels=train_Y.reshape(-1, 1),  # Reshape the labels to be 2D
        feature_names=feature_names,
        mode="regression",
        # Was originally true but you can't use a continous(numeric) values ??
        discretize_continuous=False,
        discretizer="entropy",
    )


# Create LIME explainers for Density and cP
density_explainer = create_lime_explainer(norm_train_X, train_Y[0], feature_names)
cp_explainer = create_lime_explainer(norm_train_X, train_Y[1], feature_names)


# Define a function to predict using the neural network model
def predict_fn(inputs, output_index):
    density_pred, cp_pred = model.predict(inputs)
    return np.column_stack((density_pred, cp_pred))[:, output_index]


# Select a sample case from the test set
row = 345
sample = norm_test_X[row]

# Generate explanations for Density and cP
density_exp = density_explainer.explain_instance(sample, lambda x: predict_fn(x, 0))
cp_exp = cp_explainer.explain_instance(sample, lambda x: predict_fn(x, 1))

# Plot the explanations
den_plot = density_exp.as_pyplot_figure()
plt.savefig("Data PNGs/densityLimeExplanation")
plt.show()
visc_plot = cp_exp.as_pyplot_figure()
plt.savefig("Data PNGs/viscosityLimeExplanation")
plt.show()

# Generate Shapley values and plots
# replaced norm_train_X w/ sample
# Try calculating density and viscosity shapely values separately
# It expects a vector, but we're giving it a matrix so focusing on each
# individual element might fix that

print(f'Size: {norm_val_X.shape}')
df = pd.DataFrame(norm_train_X)
column_1 = df.iloc[:, 0]
column_2 = df.iloc[:, 1]
column_3 = df.iloc[:, 2]
column_4 = df.iloc[:, 3]

visc_explainer = shap.Explainer(viscosity_model, norm_val_X)
visc_shap_values = visc_explainer(norm_val_X)
viscFig = shap.plots.bar(visc_shap_values, show=False)
plt.savefig("Data PNGs/viscosityShapleyValues")
plt.show()

den_explainer = shap.Explainer(density_model, norm_val_X)
den_shap_values = den_explainer(norm_val_X)
denFig = shap.plots.bar(den_shap_values, show=False)
plt.savefig("Data PNGs/densityShapleyValues")
plt.show()

