import keras, os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

x_cat_train = pd.read_csv("workshop_3/data/x_train_create_cat_sine_data.csv",header=None)
y_cat_train = pd.read_csv("workshop_3/data/y_train_create_cat_sine_data.csv",header=None)

x_cat_test = pd.read_csv("workshop_3/data/x_test_create_cat_sine_data.csv",header=None)
y_cat_test = pd.read_csv("workshop_3/data/y_test_create_cat_sine_data.csv",header=None)

n_neurons = 16
num_epochs = 100

encoder = OneHotEncoder(sparse_output=False)
y_cat_train_OH = encoder.fit_transform(y_cat_train)
y_cat_test_OH = encoder.fit_transform(y_cat_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(n_neurons,activation="relu"),
    tf.keras.layers.Dense(y_cat_train_OH.shape[1],activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.fit(
    x_cat_train,
    y_cat_train_OH,
    epochs=num_epochs,
    validation_data=(
        x_cat_test,
        y_cat_test_OH
    ),
    verbose=1
)

if not os.path.exists("outputs/workshop_3"):
    os.makedirs("outputs/workshop_3")

model.save("outputs/workshop_3/save_categorical_model.keras")
loaded_model = tf.keras.models.load_model("outputs/workshop_3/save_categorical_model.keras")

preds = loaded_model.predict(x_cat_test) 

n_correct = 0
avg_cert = 0
for i, val in enumerate(preds):
    if np.argmax(val) == np.argmax(y_cat_test_OH[i]):
        n_correct += 1
        avg_cert += preds[i][np.argmax(val)] * 100
    print(
        "Predicted: ",np.argmax(val),
        " Correct: ",np.argmax(y_cat_test_OH[i]),
        " Certainty: ",preds[i][np.argmax(val)] * 100,"%"
    )

print("Total accuracy: ",(n_correct/len(preds))*100,"%")
print("Average certainty of correct results: ",avg_cert/n_correct,"%")
