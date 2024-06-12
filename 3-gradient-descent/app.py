import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def gradeint_descent(x,y):
    m, b = 0, 0
    learningrate = 0.0002
    n = len(x)
    iteration = 1500000
    # iteration = 415551
    cost_previous = 0
    # m_previous = 0
    # b_previous = 0

    for i in range(iteration):
        y_predicted = m*x + b
        cost = (1/n) * sum(yi**2 for yi in (y-y_predicted))
        dm = -(2/n) * sum(x*(y-y_predicted))
        db = -(2/n) * sum(y-y_predicted)
        m = m - learningrate*dm
        b = b - learningrate*db
        print("coefficent {}, intersect {}, mse {}, iteration {} ".format( m , b , cost, i))
        # if math.isclose(cost, cost_previous, rel_tol=1e-20) and math.isclose(m, m_previous, rel_tol=1e-20) and math.isclose(b, b_previous, rel_tol=1e-20):
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        # m_previous = m
        # b_previous = b

    return m, b

def predict_using_sklean(df):
    r = LinearRegression()
    r.fit(df[['math']],df.cs)

    plt.xlabel('Math')
    plt.ylabel('Computer Science')
    plt.scatter(df.math, df.sc)
    # plt.f

    return r.coef_, r.intercept_

if __name__ == '__main__':
    df = pd.read_csv('data/test_scores.csv')
    print(df.head())

    x = np.array(df['math'])
    y = np.array(df['cs'])

    print(x)
    print(y)

    m, b = gradeint_descent(x,y)
    print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

    m_sklearn, b_sklearn = predict_using_sklean(df)
    print("Using sklearn: Coef {} Intercept {}".format(m_sklearn,b_sklearn))
