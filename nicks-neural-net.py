from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import losses
from keras import metrics
from keras import Model
from keras.callbacks import History

import matplotlib.pyplot as plt

import numpy
from numpy import ndarray

import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split

import tensorflow as tf

DATASET_PATH = "new_vehicles.csv"
IMAGE_DIR = "images/"

LEARNING_RATE = 0.001
STEPS_PER_EXECUTION = 100
BATCH_SIZE = 32
EPOCHS = 50


def create_model() -> Model:
    
    model = Sequential(
        [
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid", name = "output"),
        ]
    )

    model.compile(
        optimizer = optimizers.Adam(
            learning_rate = LEARNING_RATE
        ),
        loss = losses.MeanAbsolutePercentageError(),
        steps_per_execution = STEPS_PER_EXECUTION,
        metrics = [
            metrics.MeanAbsoluteError(),
        ],
    )

    return model


def plot_loss(history: History):
    plt.clf()
    loss_values = history.history["loss"]
    epochs = range(0, EPOCHS)
    plt.plot(scalex = epochs, scaley = loss_values, label = "Training Loss per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
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
    
    df: DataFrame = pd.read_csv(DATASET_PATH)

    y = df["price"].values
    X = df.drop("price", axis = "columns").values
    labels = df.drop("price", axis = "columns").columns

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test= train_test_split(X_scaled, y, test_size = 0.3)

    model = create_model()

    history = model.fit(
        X_train[0:100],
        tf.cast(y_train[0:100], tf.float32),
        batch_size = BATCH_SIZE, 
        epochs = EPOCHS,
    )

    y_pred = model(X_test)

    plot_loss(history)
    plot_results(y_test, y_pred)

    model.summary()