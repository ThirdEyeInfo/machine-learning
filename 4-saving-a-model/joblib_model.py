import pandas as pd
import joblib as jl
import os
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    df = pd.read_csv("data/canada_per_capita_income.csv")
    X = df.year
    y = df['per capita income (US$)']

    reg = LinearRegression()
    reg.fit(X.values.reshape(-1,1), y)

    print(reg.coef_)
    print(reg.intercept_)

    print(reg.predict([[2025]]))

    if not os.path.exists('model'):
        os.makedirs('model')

    jl.dump(reg, 'model/joblib_model')

    jl_reg = jl.load('model/joblib_model')

    print(jl_reg.predict([[2023]]))
       