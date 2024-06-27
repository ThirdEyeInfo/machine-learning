'''
Exercise
Download employee retention dataset from here: https://www.kaggle.com/giripujar/hr-analytics.

1>  Now do some exploratory data analysis to figure out which variables have direct and clear impact on 
    employee retention (i.e. whether they leave the company or continue to work)
2>  Plot bar charts showing impact of employee salaries on retention
3>  Plot bar charts showing corelation between department and employee retention
4>  Now build logistic regression model using variables that were narrowed down in step 1
5>  Measure the accuracy of the model

Refer: https://github.com/codebasics/py/tree/master/ML/7_logistic_reg/Exercise
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

'''https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter'''
reg = LogisticRegression(max_iter=1500)
reg.fit(X_train, y_train, )
print("Score ", reg.score(X_test, y_test))
print("Predicted Result ", reg.predict(X_test))
print("Actual Result ", y_test)

'''-----------------------------------Data Visualization---------------------------------------'''

'''--------------------------Data Visualization Using Train Data-------------------------------'''

'''left dependent on satisfaction_level'''
pd.crosstab(df.satisfaction_level,df.left).plot(kind='bar')
'''left independent of last_evaluation'''
pd.crosstab(df.last_evaluation,df.left).plot(kind='bar')
'''left independent of number_project'''
pd.crosstab(df.number_project,df.left).plot(kind='bar')
'''left dependent on average_montly_hours'''
pd.crosstab(df.average_montly_hours,df.left).plot(kind='bar')
'''left independent of time_spend_company'''
pd.crosstab(df.time_spend_company,df.left).plot(kind='bar')
'''left independent of Work_accident'''
pd.crosstab(df.Work_accident,df.left).plot(kind='bar')
'''left dependent on promotion_last_5years'''
pd.crosstab(df.promotion_last_5years,df.left).plot(kind='bar')
'''left independent of Department'''
pd.crosstab(df.Department,df.left).plot(kind='bar')
'''left dependent on salary'''
pd.crosstab(df.salary,df.left).plot(kind='bar')

# plt.show()

'''Let's ignore all the independent columns from dataset and evalute our prediction on 'left' column'''
X = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
y = df['left'].values

salary_dummies = pd.get_dummies(X.salary, prefix='salary')

X = pd.concat([X, salary_dummies], axis='columns')

X = X.drop(['salary', 'salary_medium'], axis='columns').values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

reg = LogisticRegression(max_iter=1500)
reg.fit(X_train, y_train, )
print("Score ", reg.score(X_test, y_test))
print("Predicted Result ", reg.predict(X_test))
print("Actual Result ", y_test)


'''
    Note: We ran model traning with without data visualization as well as with visualization. 
         The purpose of this action can be justify by observing the score of each training model. 
         Since, the given dataset is very low so, the score of each training model giving very 
         negligible improvement after data visulization. But key note over here is always analyze
         your dataset thoughly using data visulatization to imporove your model training. Other than 
         data visualization, actions such as outlier detection, duplicate data removal, gap in dataset
         are necessary to improve the model training.'''