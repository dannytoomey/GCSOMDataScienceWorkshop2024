import numpy as np
import matplotlib.pyplot as plt

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def create_lin_sine_data(noise: float, plot = False, split = None):
    x1 = np.arange(0, 3*np.pi, 0.1)
    x2 = np.random.uniform(low=1-noise, high=1+noise, size=x1.shape) 
    y_out = np.sin(x1) * x2
    if plot:
        plt.scatter(x1,y_out)
        plt.show()
    x_out = np.stack([x1,x2],axis=1) 
    if split is not None:
        x_train, x_test = x_out[:int(split*len(x_out)),:], x_out[int(split*len(x_out)):,:]
        y_train, y_test = y_out[:int(split*len(y_out))], y_out[int(split*len(y_out)):]
        np.savetxt("workshop_3/data/x_train_create_lin_sine_data.csv", x_train, delimiter=",") 
        np.savetxt("workshop_3/data/x_test_create_lin_sine_data.csv", x_test, delimiter=",") 
        np.savetxt("workshop_3/data/y_train_create_lin_sine_data.csv", y_train, delimiter=",") 
        np.savetxt("workshop_3/data/y_test_create_lin_sine_data.csv", y_test, delimiter=",") 
    else: 
        np.savetxt("workshop_3/data/x_create_lin_sine_data.csv", x_out, delimiter=",") 
        np.savetxt("workshop_3/data/y_create_lin_sine_data.csv", y_out, delimiter=",") 
    
def create_cat_sine_data(noise: float, plot = False, split = None):
    x1 = np.arange(0, 3*np.pi, 0.1)
    noise = noise
    x2 = np.random.uniform(low=1-noise, high=1+noise, size=x1.shape) 
    y = np.sin(x1) * x2
    if plot:
        plt.scatter(x1,y) 
        plt.show() 
    x_out = np.stack([x1,x2],axis=1) 
    y_out = np.zeros_like(y) 
    for i, _ in enumerate(y):
        if -max(abs(y)) <= y[i] < -max(abs(y))/2:
            y_out[i] = 1
        if -max(abs(y))/2 <= y[i] < 0.0:
            y_out[i] = 2
        if 0.0 <= y[i] < max(abs(y))/2:
            y_out[i] = 3
        if max(abs(y))/2 <= y[i] <= max(abs(y)):
            y_out[i] = 4
    if split is not None:
        x_out,y_out = unison_shuffled_copies(x_out,y_out) 
        x_train, x_test = x_out[:int(split*len(x_out)),:], x_out[int(split*len(x_out)):,:]
        y_train, y_test = y_out[:int(split*len(y_out))], y_out[int(split*len(y_out)):]
        np.savetxt("workshop_3/data/x_train_create_cat_sine_data.csv", x_train, delimiter=",") 
        np.savetxt("workshop_3/data/x_test_create_cat_sine_data.csv", x_test, delimiter=",") 
        np.savetxt("workshop_3/data/y_train_create_cat_sine_data.csv", y_train, delimiter=",") 
        np.savetxt("workshop_3/data/y_test_create_cat_sine_data.csv", y_test, delimiter=",") 
    else: 
        np.savetxt("workshop_3/data/x_create_cat_sine_data.csv", x_out, delimiter=",") 
        np.savetxt("workshop_3/data/y_create_cat_sine_data.csv", y_out, delimiter=",") 
    