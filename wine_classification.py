import pandas as pd
dataset = pd.read_csv('https://huawei.cpsoftware.com.br/hcia-ai-7903bc60-1f7f-4799-b7de-e56e2ff7ffd5-wine-dataset.csv')

# Dataset - In collab this command show the dataset

dataset.head()

# Preparing the data

#Change the values of the style column to a numerical value
dataset['style'] = dataset['style'].replace('red', 0)

dataset['style'] = dataset['style'].replace('white', 1)

# The predictor 

y = dataset['style']

x = dataset.drop('style', axis=1)

# Data traning and Data Testing

from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.3)

# Creating the traning model

from sklearn.ensemble import ExtraTreesClassifier

# Model Creation

model = ExtraTreesClassifier()

model.fit(x_train, y_train)


# Cheaking the accuracy of the algorithm
result = model.score(x_test, y_test)

print("Accuracy", result)

# This code filters some dataset samples so we can check the forecasts
# This filtering will only contain the type of wine

y_test[300:303]

# This code filters some samples of the dataset and it is in it that
# all the characteristics of the wine will be so that our algorithm finds out if it is red or white
x_test[400:403]

forecasts = model.predict(x_test[300:303])

print(forecasts)


