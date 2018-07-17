# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')
X = dataset.iloc[:,[2,3,4,5,6,7,8,9,10]].values
Xtest = testset.iloc[:,[2,3,4,5,6,7,8,9,10]].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 7:])
X[:, 7:] = imputer.transform(X[:, 7:])

imputer1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer1 = imputer1.fit(Xtest[:, 7:])
Xtest[:, 7:] = imputer1.transform(Xtest[:, 7:])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 3] = labelencoder_X2.fit_transform(X[:, 3])
labelencoder_X3 = LabelEncoder()
X[:, 4] = labelencoder_X3.fit_transform(X[:, 4])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Xt = LabelEncoder()
Xtest[:, 0] = labelencoder_Xt.fit_transform(Xtest[:, 0])
labelencoder_Xt1 = LabelEncoder()
Xtest[:, 1] = labelencoder_Xt1.fit_transform(Xtest[:, 1])
labelencoder_Xt2 = LabelEncoder()
Xtest[:, 3] = labelencoder_Xt2.fit_transform(Xtest[:, 3])
labelencoder_Xt3 = LabelEncoder()
Xtest[:, 4] = labelencoder_Xt3.fit_transform(Xtest[:, 4])
onehotencoder1 = OneHotEncoder(categorical_features = [3])
Xtest = onehotencoder1.fit_transform(Xtest).toarray()
# Avoiding the Dummy Variable Trap
X = X[:, 1:]
Xtest = Xtest[:, 1:]

from sklearn.ensemble import RandomForestRegressor
regressor123 = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor123.fit(X, y)


y_predf = regressor123.predict(Xtest)