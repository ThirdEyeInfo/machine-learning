'''
Exercise
Download employee retention dataset from here: https://www.kaggle.com/giripujar/hr-analytics.

1>  Now do some exploratory data analysis to figure out which variables have direct and clear impact on 
    employee retention (i.e. whether they leave the company or continue to work)
2>  Plot bar charts showing impact of employee salaries on retention
3>  Plot bar charts showing corelation between department and employee retention
4>  Now build logistic regression model using variables that were narrowed down in step 1
5>  Measure the accuracy of the model
'''
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('data\HR_comma_sep.csv')



le_department = LabelEncoder()
le_salary = LabelEncoder()

df['department_n'] = le_department.fit_transform(df['Department'])
df['salary_n'] = le_salary.fit_transform(df['salary'])

X = df.drop(['left', 'salary', 'Department'], axis='columns').values
y = df['left'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print(X_train)
print(y_train)

'''https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter'''
reg = LogisticRegression(max_iter=15000)
reg.fit(X_train, y_train, )
print(reg.score(X_test, y_test))

print(reg.predict(X_test))
print(y_test)

'''-----------------------------------Data Visualization---------------------------------------'''

'''--------------------------Data Visualization Using Train Data-------------------------------'''
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.hist(X_train)
plt.plot(X_train, y_train)