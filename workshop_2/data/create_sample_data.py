import numpy as np

def create_vote_data(n_samples: np.uint16, n_classes: np.uint16, split = None):
    x_out = (np.random.rand(n_samples,n_classes)*100).astype(np.float32) 
    y_out = np.argmax(x_out,axis=1).astype(np.int16) 
    if split is not None:
        x_train, x_test = x_out[:int(split*len(x_out)),:], x_out[int(split*len(x_out)):,:]
        y_train, y_test = y_out[:int(split*len(y_out)),:], y_out[int(split*len(y_out)):,:]
        np.savetxt("workshop_2/data/x_train_create_vote_data.csv", x_train, delimiter=",") 
        np.savetxt("workshop_2/data/x_test_create_vote_data.csv", x_test, delimiter=",") 
        np.savetxt("workshop_2/data/y_train_create_vote_data.csv", y_train, delimiter=",") 
        np.savetxt("workshop_2/data/y_test_create_vote_data.csv", y_test, delimiter=",") 
    else: 
        np.savetxt("workshop_2/data/x_create_vote_data.csv", x_out, delimiter=",") 
        np.savetxt("workshop_2/data/y_create_vote_data.csv", y_out, delimiter=",") 

def create_linear_data(n_samples: np.uint16, n_classes: np.uint16, split = None):
    x_out = (np.random.rand(n_samples,n_classes)*100).astype(np.float32)
    y_out = np.sum(x_out,axis=1)
    if split is not None:
        x_train, x_test = x_out[:int(split*len(x_out)),:], x_out[int(split*len(x_out)):,:]
        y_train, y_test = y_out[:int(split*len(y_out)),:], y_out[int(split*len(y_out)):,:]
        np.savetxt("workshop_2/data/x_train_create_linear_data.csv", x_train, delimiter=",") 
        np.savetxt("workshop_2/data/x_test_create_linear_data.csv", x_test, delimiter=",") 
        np.savetxt("workshop_2/data/y_train_create_linear_data.csv", y_train, delimiter=",") 
        np.savetxt("workshop_2/data/y_test_create_linear_data.csv", y_test, delimiter=",") 
    else: 
        np.savetxt("workshop_2/data/x_create_linear_data.csv", x_out, delimiter=",") 
        np.savetxt("workshop_2/data/y_create_linear_data.csv", y_out, delimiter=",") 
