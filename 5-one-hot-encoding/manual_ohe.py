import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('data/carprices.csv')

dummies = pd.get_dummies(df['Car Model'])
merged = pd.concat([df,dummies], axis='columns')
merged = merged.drop(['Car Model','Mercedez Benz C class'], axis='columns')

X = merged.drop(['Sell Price($)'], axis='columns')
y = merged['Sell Price($)']

merged.plot.scatter(x='Sell Price($)', y='Mileage', s=100, c='r')
merged.plot.scatter(x='Sell Price($)', y='Age(yrs)', s=100, c='b')
plt.show()

reg = LinearRegression()
reg.fit(X, y)
print(reg.score(X, y))
print(reg.predict([[35000, 5, True, False]]))
