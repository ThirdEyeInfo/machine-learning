import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('data/insurance_data.csv')

X = df.drop(['bought_insurance'], axis='columns').values
y = df['bought_insurance'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

reg = LogisticRegression()
reg.fit(X_train, y_train)
reg.score(X_test, y_test)

res = reg.predict(X_test)
print("Predicted result: ", res)
print("Actual result: ", y_test)
"""-----------------------------------Data Visualization---------------------------------------"""
def sortArray(arr):
    arr = np.array(arr).flatten()
    arr = np.sort(arr)
    return arr

def sigmoid(X):
    a = []
    for item in X:
        '''             S(x) = 1/1+e^-x 
            Where S(x) = Sigmoid function to draw S shape curve in a range of 0 to 1 as binary distribution
                    x = input value to sigmoid function ranging -ve infinitite to +ve infinitite
                    e = Euler's constant i.e. ~2.71828
        '''
        '''if the value of the Sigmoid function S(x) for an input is greater than 0.5 (threshold value)
          then it belongs to Class 1 otherwise it belongs to Class 0'''
        prediction = reg.predict([[item]])[0]
        a.append(1 if (1/(1 + math.exp(-prediction))) > 0.5 else 0)
    return (a)

'''Derivative of Sigmoid function'''
## def d_sigmoid(x):
##     a = []
##     list = sigmoid(x)
##     for sig in list:
##         a.append(sig * (1-sig))
##     return  a
"""-----------------------------------Data Visualization Using Train Data---------------------------------------"""
# print('=======================================================================')
X_train = sortArray(X_train)
# print(X_train)
V = sigmoid(X_train)
# print('=======================================================================')
# print(V)

plt.xlabel('Age')
plt.ylabel('Insurance Flag')
plt.scatter(X_train, y_train, marker='+', color='red')
plt.plot(X_train, V)
plt.grid(True, linestyle='--')
## df = pd.DataFrame({"x": X_train, "sigmoid(x)": V, "d_sigmoid(x)": d_sigmoid(X_train)})
# print(df)
plt.show()

"""-----------------------------------Data Visualization Using Test Data---------------------------------------"""
# print('=======================================================================')
X_test = sortArray(X_test)
print(X_test)
V = sigmoid(X_test)
# print('=======================================================================')
# print(V)

plt.xlabel('Age')
plt.ylabel('Insurance Flag', labelpad=0.1)
plt.scatter(X_test, y_test, marker='+', color='red')
plt.plot(X_test, V)
plt.grid(True, linestyle='--')
## df = pd.DataFrame({"x": X_test, "sigmoid(x)": V, "d_sigmoid(x)": d_sigmoid(X_test)})
# print(df)
plt.show()

"""-----------------------------------Data Visualization With All DataSet---------------------------------------"""

# print('=======================================================================')
X = sortArray(X)
# print(X)
V = sigmoid(X)
# print('=======================================================================')
# print(V)

plt.xlabel('Age')
plt.ylabel('Insurance Flag')
plt.scatter(X, y, marker='+', color='red')
plt.plot(X, V)
plt.grid(True, linestyle='--')
## df = pd.DataFrame({"x": X, "sigmoid(x)": V, "d_sigmoid(x)": d_sigmoid(X)})
# print(df)
plt.show()