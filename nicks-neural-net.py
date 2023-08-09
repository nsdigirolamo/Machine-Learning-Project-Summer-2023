from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, initializers, regularizers
from keras import optimizers, initializers, regularizers
from keras import losses
from keras import metrics
from keras import Model
from keras.callbacks import History

import dill

import dill

import matplotlib.pyplot as plt

import numpy as np
from numpy import ndarray

import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

import tensorflow as tf

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

DATASET_PATH = "new_vehicles.csv"
IMAGE_DIR = "images/"

LEARNING_RATE = 0.0001
STEPS_PER_EXECUTION = 1
BATCH_SIZE = 512
EPOCHS = 10
VALIDATION_SPLIT = 0.2

SCALER = StandardScaler()
LOSS = losses.MeanSquaredError()
OPTIMIZER = optimizers.RMSprop(
    learning_rate = LEARNING_RATE
)
METRICS = [ 
    metrics.MeanAbsoluteError(),
]


def create_model() -> Model:
    
    model = Sequential(
        [
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1, activation="relu", name = "output"),
        ]
    )

    model.compile(
        optimizer = OPTIMIZER,
        loss = LOSS,
        steps_per_execution = STEPS_PER_EXECUTION,
        metrics = METRICS
    )

    return model


def plot_loss(history: History):
    loss_values = history.history['loss']
    loss_values = history.history['loss']
    epochs = range(0, EPOCHS)
    plt.clf()
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.clf()
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{IMAGE_DIR}/loss_per_epoch.png")


def plot_results(y_actual: ndarray, y_pred: ndarray):
    plt.clf()
    plt.scatter(x = y_actual, y = y_pred, label = "Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.axis("equal")
    plt.savefig(f"{IMAGE_DIR}/actual_vs_predicted.png")


if __name__ == "__main__":
    
    df: DataFrame = pd.read_csv(DATASET_PATH, encoding = "utf8")

    labels = df.columns
    labels = df.columns
    y = df["price"].values
    X = df.drop("price", axis = "columns").values

    ct = ColumnTransformer(
        [
            ("scaler", SCALER, [0, 1]), # scale only the odometer and year columns
            ("passthrough", "passthrough", slice(2, -1)), # Makes sure the categorical columns dont get dropped
        ],
    )

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.3, random_state = 42)

    ct.fit(X_train)
    X_train_scaled = ct.transform(X_train)
    X_test_scaled = ct.transform(X_test)
    X_train_scaled = np.asarray(X_train_scaled).astype("float32")
    X_test_scaled = np.asarray(X_test_scaled).astype("float32")

    print(f"X_train shape: {X_train.shape}")
    print(f"First element of X_train {X_train[0]}")

    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"First element of X_train_scaled {X_train_scaled[0]}")

    model = create_model()

    history = model.fit(
          X_train_scaled,
          tf.cast(y_train, tf.float32),
          batch_size = BATCH_SIZE, 
          epochs = EPOCHS,
          validation_split = VALIDATION_SPLIT,
    )

    with open("dilledmodel", "wb") as dill_file:
        dill.dump(model, dill_file)

    y_pred = model.predict(X_test_scaled).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values [Price]')
    plt.ylabel('Predictions [Price]')
    lims = [0, 100000]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.savefig(f"{IMAGE_DIR}/scatter.png")

    plot_loss(history)

    y_pred = model.predict(X_test_scaled)

    plot_results(y_test, y_pred)

    # model.summary()