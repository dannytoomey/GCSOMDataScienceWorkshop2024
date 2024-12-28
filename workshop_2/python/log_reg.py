# https://towardsdatascience.com/multiclass-logistic-regression-from-scratch-9cc0007da372
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
onehot_encoder = OneHotEncoder(sparse_output=False)

def loss(X, Y, W):
    """
    Y: onehot encoded
    """
    Z = - X @ W
    N = X.shape[0]
    loss = 1/N * (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
    return loss

def gradient(X, Y, W, mu):
    """
    Y: onehot encoded 
    """
    Z = - X @ W
    P = softmax(Z, axis=1)
    N = X.shape[0]
    gd = 1/N * (X.T @ (Y - P)) + 2 * mu * W
    return gd

def gradient_descent(X, Y, max_iter=1000, eta=0.1, mu=0.01):
    """
    Very basic gradient descent algorithm with fixed eta and mu
    """
    Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))
    W = np.zeros((X.shape[1], Y_onehot.shape[1]))
    step = 0
    step_lst = [] 
    loss_lst = []
    W_lst = []
 
    while step < max_iter:
        step += 1
        W -= eta * gradient(X, Y_onehot, W, mu)
        step_lst.append(step)
        W_lst.append(W)
        loss_lst.append(loss(X, Y_onehot, W))

    df = pd.DataFrame({
        'step': step_lst, 
        'loss': loss_lst
    })
    return df, W

class Multiclass:
    def fit(self, X, Y):
        self.loss_steps, self.W = gradient_descent(X, Y)

    def predict(self, H, show_softmax = False):
        Z = - H @ self.W
        P = softmax(Z, axis=1)
        if show_softmax:
            print(P[0][np.argmax(P, axis=1)])
        return np.argmax(P, axis=1)

x_cat = pd.read_csv("workshop_2/data/x_create_vote_data.csv",header=None)
y_cat = pd.read_csv("workshop_2/data/y_create_vote_data.csv",header=None)

# fit model
model = Multiclass()
model.fit(x_cat.to_numpy(), y_cat.to_numpy())

# predict 
new_x = np.array([[31,32,33]])
print("Prediction: ",model.predict(new_x)) 
print("Correct: ",np.argmax(new_x))
