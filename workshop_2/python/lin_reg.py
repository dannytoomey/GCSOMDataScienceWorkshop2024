import numpy as np
import pandas as pd

x_lin = pd.read_csv("workshop_2/data/x_create_linear_data.csv",header=None)
y_lin = pd.read_csv("workshop_2/data/y_create_linear_data.csv",header=None)

# append ones column to find intercept
# https://stackoverflow.com/questions/45061326/how-we-can-compute-intercept-and-slope-in-statsmodels-ols
x_lin = pd.concat([x_lin, pd.DataFrame(np.ones_like(x_lin[0]))], axis=1)

# https://jianghaochu.github.io/ordinary-least-squares-regression-in-python-from-scratch.html
beta_hat = np.array(np.matmul(
    np.matmul(np.linalg.inv(np.matmul(np.array(x_lin).transpose(), 
    np.array(x_lin))), 
    x_lin.transpose()), 
    y_lin
))

def predict(x_new,beta_hat):
    pred = 0
    for i in range(0,len(x_new)):
        pred += x_new[i] * beta_hat[i]
    pred += beta_hat[-1]
    return pred

print("Coefficients: ",beta_hat[:-1])
print("Intercept: ",beta_hat[-1]) 

x_new = [30,40,50]
print("Prediction: ",predict(x_new,beta_hat))
print("Correct: ",sum(x_new))