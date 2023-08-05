import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import tensorflow as tf


DATA_DIR = "new_vehicles.csv"

if __name__ == "__main__":
    # Read in dataset
    vehicles = pd.read_csv(DATA_DIR)
    # Convert to more appropriate types
    dataset = vehicles.convert_dtypes()

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_labels = train_dataset.pop('price')
    test_labels = test_dataset.pop('price')

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=[len(train_dataset.keys())]),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])

    print(model.summary)

    EPOCHS = 10

    history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()])

    plotter.plot({'Basic': history}, metric = "mae")
    plt.ylim([0, 5000])
    plt.ylabel('MAE [Price]')


    

