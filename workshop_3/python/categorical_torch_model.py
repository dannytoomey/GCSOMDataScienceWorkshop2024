import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim

torch.xpu.is_available() 

x_cat_train = torch.tensor(
    np.loadtxt("workshop_3/data/x_train_create_cat_sine_data.csv", delimiter=","), 
    dtype=torch.float32
)
y_cat_train = torch.tensor(
    np.loadtxt("workshop_3/data/y_train_create_cat_sine_data.csv", delimiter=","),
    dtype=torch.int32
)
y_cat_train = (y_cat_train - 1).type(torch.LongTensor)

x_cat_test = torch.tensor(
    np.loadtxt("workshop_3/data/x_test_create_cat_sine_data.csv", delimiter=","),
    dtype=torch.float32
)
y_cat_test = torch.tensor(
    np.loadtxt("workshop_3/data/y_test_create_cat_sine_data.csv", delimiter=","),
    dtype=torch.int32
)
y_cat_test = (y_cat_test - 1).type(torch.LongTensor)

class CategoricalModel(nn.Module):
    def __init__(self, x_n_features, hidden_layer_size, y_n_features):
        super().__init__()
        self.input_layer = nn.Linear(x_n_features, hidden_layer_size)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_layer_size, y_n_features)
        self.output_activation = nn.Softmax()
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.output_activation(self.output_layer(x))
        return x

x_n_features = x_cat_train.shape[1]
hidden_layer_size = 16
y_n_features = max(y_cat_train)+1

model = CategoricalModel(x_n_features, hidden_layer_size, y_n_features)
loss_fn = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
print_fraction = 10
print_every = n_epochs / print_fraction
count = 0

for epoch in range(1,n_epochs+1):
    y_pred = model.forward(x_cat_train)
    loss = loss_fn(y_pred, y_cat_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    count += 1
    if count == print_every:
        val_preds = torch.argmax(model.forward(x_cat_test),dim=1)
        val_acc = 0
        for i,pred in enumerate(val_preds):
            if pred == y_cat_test[i]:
                val_acc += 1
        
        print(f"Epoch: {epoch}, Training loss: {loss}, Val accuracy: {val_acc/len(val_preds)}")
        count = 0

torch.save(model.state_dict(), "outputs/workshop_3/save_torch_categorical_model.pth")
loaded_model = CategoricalModel(x_n_features, hidden_layer_size, y_n_features)
loaded_model.load_state_dict(torch.load("outputs/workshop_3/save_torch_categorical_model.pth"))

preds = torch.argmax(loaded_model.forward(x_cat_test),dim=1)
vals = 0
for i,pred in enumerate(val_preds):
    if pred == y_cat_test[i]:
        vals += 1

print("Validation accuracy:",vals/len(preds))
