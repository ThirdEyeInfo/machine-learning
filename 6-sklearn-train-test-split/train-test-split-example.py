import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/carprices-v2.csv')

X = df.drop(['Sell Price($)'], axis='columns').values
y = df['Sell Price($)'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

reg = LinearRegression()
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))
print(reg.predict(X_test))
print(y_test)