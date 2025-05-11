import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim

torch.xpu.is_available() 

x_lin_train = torch.tensor(
    np.loadtxt("workshop_3/data/x_train_create_lin_sine_data.csv", delimiter=","), 
    dtype=torch.float32
)
y_lin_train = torch.tensor(
    np.loadtxt("workshop_3/data/y_train_create_lin_sine_data.csv", delimiter=","),
    dtype=torch.float32
).reshape(-1, 1)

x_lin_test = torch.tensor(
    np.loadtxt("workshop_3/data/x_test_create_lin_sine_data.csv", delimiter=","),
    dtype=torch.float32
)
y_lin_test = torch.tensor(
    np.loadtxt("workshop_3/data/y_test_create_lin_sine_data.csv", delimiter=","),
    dtype=torch.float32
).reshape(-1, 1)

class LinearModel(nn.Module):
    def __init__(self, x_n_features, hidden_layer_size, y_n_features):
        super().__init__()
        self.input_layer = nn.Linear(x_n_features, hidden_layer_size)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_layer_size, y_n_features)
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.output_layer(x)
        return x

x_n_features = x_lin_train.shape[1]
hidden_layer_size = 16
y_n_features = 1

model = LinearModel(x_n_features, hidden_layer_size, y_n_features)
loss_fn = nn.L1Loss() # AKA mean absolute error loss 
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
print_fraction = 10
print_every = n_epochs / print_fraction
val_precision = 0.5
count = 0

for epoch in range(1,n_epochs+1):
    y_pred = model.forward(x_lin_train)
    loss = loss_fn(y_pred, y_lin_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    count += 1
    if count == print_every:
        val_preds = model.forward(x_lin_test)
        val_acc = 0
        for i,pred in enumerate(val_preds):
            if abs(pred - y_lin_test[i]) <= val_precision:
                val_acc += 1
        
        print(f"Epoch: {epoch}, Training loss: {loss}, Val accuracy: {val_acc/len(val_preds)}")
        count = 0

torch.save(model.state_dict(), "outputs/workshop_3/save_torch_linear_model.pth")
loaded_model = LinearModel(x_n_features, hidden_layer_size, y_n_features)
loaded_model.load_state_dict(torch.load("outputs/workshop_3/save_torch_linear_model.pth"))

preds = loaded_model.forward(x_lin_test) 
vals = 0
for i,pred in enumerate(val_preds):
    vals += float(abs(pred - y_lin_test[i]))

print("Average validation MAE:",vals/len(preds))
