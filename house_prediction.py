import pandas as pd

from sklearn import linear_model as lm

import numpy as np

df_raw = pd.read_csv('https://huawei.cpsoftware.com.br/hcia-ai-4b3d7ebd-6222-4999-aa4e-040087977f90-kc-house-data.csv')

df_raw.head()

# Data Preparation

x_train = df_raw.drop(['price', 'data'], axis=1)

y_train = df_raw['price'].copy()

# Creating Model Tranning

model_lr = lm.LinearRegression()

# Model Traning

model_lr.fit(x_train, y_train)

# Prediction

pred = model_lr.predict(x_train)


# We created a new variable with the characteristics of the houses and added the

# column prediction with the values of the variable pred
df1 = df_raw.copy()

df1['prediction'] = pred

# Now we create the error column to represent the MAE and the column error_abs to store the absolute value 
df1['error'] = df1['price'] - df1['prediction']

df1['error_abs'] = np.abs( df1['error'] )


# Likewise, we now create the error_perc column to represent MAPE and the error_perc_abs column to store the absolute value
df1['error_perc'] = ( (df1['price'] - df1['prediction']) / df1['price'] )

df1['error_perc_abs'] = np.abs( df1['error_perc'] )


# Mean absolute error
mae = np.mean( df1['error_abs'] )

print('MAE: {}'.format( mae ))


# Mean absolute percentage error
mape = np.mean( df1['error_perc_abs'] )

print('MAPE: {}'.format( mape ))
