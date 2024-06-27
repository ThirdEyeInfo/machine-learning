import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv('data\iris.csv')

X = df[['sepal_length','sepal_width','petal_length','petal_width']].values
y = df.species.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LogisticRegression(max_iter=1500)
reg.fit(X_train, y_train)
print("Score ", reg.score(X_test, y_test))
y_predicted =  reg.predict(X_test)
print("Prediction ", y_predicted)

"""Confusion matrix"""
cm = confusion_matrix(y_test, y_predicted)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()