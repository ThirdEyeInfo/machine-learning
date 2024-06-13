import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('data/carprices.csv')

dummies = pd.get_dummies(df['Car Model'])
merged = pd.concat([df,dummies], axis='columns')
merged = merged.drop(['Car Model','Mercedez Benz C class'], axis='columns')

X = merged.drop(['Sell Price($)'], axis='columns')
y = merged['Sell Price($)']

merged.plot.scatter(x='Mileage', y='Sell Price($)', s=100, c='r')
merged.plot.scatter(x='Age(yrs)', y='Sell Price($)', s=100, c='b')
plt.show()

reg = LinearRegression()
reg.fit(X.values, y.values)
print(reg.score(X.values, y.values))

res = reg.predict([[25000, 5, True, False], [50000, 5, True, False], [35000, 5, True, False]])
print("Audi A5")
print(res)

res = reg.predict([[25000, 5, False, False], [50000, 5, False, False], [35000, 5, False, False]])
print("Mercedez Benz C class")
print(res)

res = reg.predict([[25000, 5, False, True], [50000, 5, False, True], [35000, 5, False, True]])
print("BMW X5")
print(res)