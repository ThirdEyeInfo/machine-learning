import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('data/carprices.csv')

le = LabelEncoder()
df['Car Model'] = le.fit_transform(df['Car Model'])

ct = ColumnTransformer([('Car Model', OneHotEncoder(), [0])], remainder = 'passthrough')
X = df[['Car Model', 'Mileage', 'Age(yrs)']].values
X = ct.fit_transform(X)
y = df['Sell Price($)'].values

# print(X)

reg = LinearRegression()

X_mile = X[:,3:-1]
reg.fit(X_mile, y)
plt.xlabel('Mileage')
plt.ylabel('Sell Price')
plt.scatter(X_mile, df['Sell Price($)'])
plt.plot(X_mile, reg.predict(X_mile), color='red')

plt.show()

X_age = X[:,4:]
reg.fit(X_age, y)
plt.xlabel('Age')
plt.ylabel('Sell Price')
plt.scatter(X_age, df['Sell Price($)'])
plt.plot(X_age, reg.predict(X_age), color='red')

plt.show()

# X_BMW_X5 = X[:,1:-3]
# print(X_BMW_X5)
# reg.fit(X_BMW_X5, y)
# plt.xlabel('BMW X5 Model')
# plt.ylabel('Sell Price')
# plt.scatter(X_BMW_X5, df['Sell Price($)'])
# plt.plot(X_BMW_X5, reg.predict(X_BMW_X5), color='red')

# plt.show()

X = X[:,1:]

reg.fit(X, y)
res = reg.predict([[0, 0, 25000, 5],[0, 0, 50000, 5], [0, 0, 35000, 5]])
print("Audi A5")
print(res)

res = reg.predict([[0, 1, 25000, 5],[0, 1, 50000, 5], [0, 1, 35000, 5]])
print("Mercedez Benz C class")
print(res)

res = reg.predict([[1, 0, 25000, 5],[1, 0, 50000, 5], [1, 0, 35000, 5]])
print("BMW X5")
print(res)

# df.plot.scatter(x='Sell Price($)', y='Age(yrs)', c='b')
# df.plot.scatter(x='Mileage', y='Sell Price($)', c='r')
# df.plot.line(x='Mileage', y=reg.predict(X_mile))