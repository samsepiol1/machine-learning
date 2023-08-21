# Prevent unnecessary warnings.

import warnings
warnings.filterwarnings("ignore")

# Introduce the basic package of data science.
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import seaborn as sns

##Set attributes to prevent garbled characters in Chinese.
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# Introduce machine learning, preprocessing, model selection, and evaluation indicators.
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# Import the Boston dataset used this time.
from sklearn.datasets import load_boston

# Introduce algorithms.
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression, ElasticNet

# Compared with SVC, it is the regression form of SVM.
from sklearn.svm import SVR

# Integrate algorithms.
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Loading dataset, viewing data attributes and more

# Load the price data set
boston = load_boston()

# X Features and y Labels

x = boston.data
y = boston.target

# Display Attributes

print('Feature column name')
print(boston.feature_names)
print("Sample data volume: %d, number of features: %d" % x.shape)
print("Target sample data volume: %d" % y.shape[0])


# Convert dataframe format
x = pd.DataFrame(boston.data, columns=boston.features_names)
x.head(20)

# Data label distribution

sns.displot(tuple(y), kde=False, fit=st.norm)

# Split and pre-processing data set

# Segment the data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28)

# Strandardize the data set

ss = StandardScaler()
x_train = ss.fit_transform(x_train)

x_test = ss.transform(x_test)

x_train[0:100]

# Using multiple regression models to solve the problem

names = ['LinerRegression',
         'Ridge',
         'Lasso',
         'Random Forest',
         'GBDT',
         'Support Vector Regression',
         'ElasticNet',
         'XgBoost']

# Cross-Validation Ideia

models = [LinearRegression(),
          RidgeCV(alphas=(0.001, 0.1, 1), cv=3),
    LassoCV(alphas=(0.001, 0.1, 1), cv=5),
    RandomForestRegressor(n_estimators=10),
    GradientBoostingRegressor(n_estimators=30),
    SVR(),
    ElasticNet(alpha=0.001, max_iter=10000),
    XGBRegressor()]


# Scores of all regression models


# R2 Score function

def R2(model, x_train, x_test, y_train, y_test):
    model_fitted = model.fit(x_train, y_train)
    y_pred = model_fitted.predict(x_test)
    score = r2_score(y_test, y_pred)
    return score

# Traverse all models to score.
for name, model in zip(names, models):
    score = R2(model, x_train, x_test, y_train, y_test)
    print("{}: {:.6f}".format(name, score.mean()))


# Adjusting Hyperparameters by Grid Search

'''
 'kernel': kernel function
 'C': SVR regularization factor
 'gamma': 'rbf', 'poly' and 'sigmoid' kernel function coefficient, which affects the model performance
'''

parameters = {'kernel': ['linear', 'rbf'],
              'C': [0.1, 0.5,0.9,1,5],
              'gamma': [0.001,0.01,0.1,1]
             }

# use grid search and perform cross validation

model = GridSearchCV(SVR(), param_grid=parameters, cv=3)

model.fit(x_train, y_train)

print("Optimal parameter list:", model.best_params_)
print("Optimal model:", model.best_estimator_)
print("Optimal R2 value:", model.best_score_)


# Perform Visualization

ln_x_test = range(len(x_test))
y_predict = model.predict(x_test)

plt.figure(figsize=(16,8), facecolor='w')

#Draw with a red solid line.
plt.plot (ln_x_test, y_test, 'r-', lw=2, label=u'Value')

#Draw with a green solid line.
plt.plot (ln_x_test, y_predict, 'g-', lw = 3, label=u'Estimated value of the SVR algorithm, $R^2$=%.3f' %
(model.best_score_))

#Display in a diagram.
plt.legend(loc ='upper left')
plt.grid(True)
plt.title(u"Boston Housing Price Forecast (SVM)")
plt.xlim(0, 101)
plt.show()





