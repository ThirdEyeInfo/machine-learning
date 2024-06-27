import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv('data\salaries.csv')
# print(df.head)

X = df.drop('salary_more_then_100k', axis='columns')
y = df.salary_more_then_100k

# print(X)
# print(y)

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

X['company_n'] = le_company.fit_transform(df['company'])
X['job_n'] = le_job.fit_transform(df['job'])
X['degree_n'] = le_degree.fit_transform(df['degree'])

X = X.drop(['company','job','degree'], axis='columns')

# print(X)
random_state=21
test_size=0.38
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(f'Model Score with random_state : {random_state} & test_size : {test_size} is = ', model.score(X_test, y_test))
y_predicted = model.predict(X_test)
# print('Test Prediction ', y_predicted)

"""Confusion matrix"""
cm = confusion_matrix(y_test, y_predicted)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()