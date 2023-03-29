import pandas as pd
import patsy
import statsmodels.api as sm

xls = pd.ExcelFile('API_ProjectAll_SI.xlsx')
staticDenDataframe = pd.read_excel(xls, 'staticDen')
staticViscDataframe = pd.read_excel(xls, 'staticVisc')

selectStaticDen = staticDenDataframe[['T', 'ALogP', 'ALogP2', 'AMR']].copy()
selectStaticVisc = staticViscDataframe[['T', 'ALogP', 'ALogP2', 'AMR']].copy()

print("Static Density Data Analysis")
yT, X = patsy.dmatrices("T ~ ALogP + ALogP2 + AMR", data=selectStaticDen, return_type="dataframe")
X.info()
model = sm.OLS(yT, X)
results = model.fit()
print(results.summary())

print()
print(print("Static Viscosity Data Analysis"))
yT, X = patsy.dmatrices("T ~ ALogP + ALogP2 + AMR", data=selectStaticVisc, return_type="dataframe")
X.info()
model = sm.OLS(yT, X)
results = model.fit()
print(results.summary())