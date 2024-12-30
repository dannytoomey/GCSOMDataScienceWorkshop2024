import keras, os
import tensorflow as tf
import pandas as pd
import numpy as np

x_lin_train = pd.read_csv("workshop_3/data/x_train_create_lin_sine_data.csv",header=None)
y_lin_train = pd.read_csv("workshop_3/data/y_train_create_lin_sine_data.csv",header=None)

x_lin_test = pd.read_csv("workshop_3/data/x_test_create_lin_sine_data.csv",header=None)
y_lin_test = pd.read_csv("workshop_3/data/y_test_create_lin_sine_data.csv",header=None)

n_neurons = 16
num_epochs = 100

model = tf.keras.Sequential([
    tf.keras.layers.Dense(n_neurons,activation="relu"),
    tf.keras.layers.Dense(y_lin_train.shape[1])
])

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.MeanAbsoluteError(),
    metrics=['mae']
)

model.fit(
    x_lin_train,
    y_lin_train,
    epochs=num_epochs,
    validation_data=(
        x_lin_test, 
        y_lin_test
    ),
    verbose=1
)

if not os.path.exists("outputs/workshop_3"):
    os.makedirs("outputs/workshop_3")

model.save("outputs/workshop_3/save_regression_model.keras")
loaded_model = tf.keras.models.load_model("outputs/workshop_3/save_regression_model.keras")

preds = loaded_model.predict(x_lin_test) 
vals = []
for i in range(0,preds.shape[0]):
    vals.append(np.abs(y_lin_test.iloc[i]-preds[i]))

print("Average validation MAE:",np.mean(vals))
