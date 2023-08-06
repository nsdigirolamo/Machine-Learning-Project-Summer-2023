from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import losses
from keras import metrics

import numpy
from numpy import ndarray

import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Filepaths
DATASET_PATH = "new_vehicles.csv"

# Neural Network Constants
LEARNING_RATE = 0.001


if __name__ == "__main__":
    
    df: DataFrame = pd.read_csv(DATASET_PATH)

    y = df["price"].values
    X = df.drop("price", axis = "columns").values
    labels = df.drop("price", axis = "columns").columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    """
    X_training = 70% of labels
    X_validation = 15% of labels
    X_testing = 15% of labels
    y_training = 70% of prices
    y_validation = 15% of prices
    y_testing = 15% of prices
    """
    X_training, X_testing_and_validation, y_training, y_testing_and_validation = train_test_split(X_scaled, y, test_size = 0.3)
    X_validation, X_testing, y_validation, y_testing = train_test_split(X_testing_and_validation, y_testing_and_validation, test_size = 0.5)

    model = Sequential(
        [
            Dense(256, activation="relu", input_shape = (94, ), name = "layer1"),
            Dense(128, activation="relu", name = "layer3"),
            Dense(64, activation="relu", name = "layer5"),
            Dense(1, activation="sigmoid", name = "output"),
        ]
    )

    model.compile(
        optimizer = optimizers.Adam(learning_rate = LEARNING_RATE),
        loss = losses.MeanSquaredError(),
        metrics = [
            metrics.Accuracy(),
            metrics.MeanSquaredError(),
        ],
    )

    hist = model.fit(
        X_training,
        y_training,
        batch_size = 32, 
        epochs = 10,
        validation_data = (X_validation, y_validation)
    )

    model.summary()