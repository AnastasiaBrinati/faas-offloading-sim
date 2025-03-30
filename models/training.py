import pandas as pd
import tensorflow as tf
import numpy as np
import random
import keras_tuner as kt
from tensorflow import keras
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys

# Set random seed for reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Initialize a MinMaxScaler
SCALER = MinMaxScaler(feature_range=(0, 1))
BATCH_SIZE = 8
SEQ_LENGTH = 7


# Function to load dataset from a CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Function to prepare training, validation, and test datasets
def prepare_data(df, test_size=0.35):
    rates = df.copy()
    l = int(len(rates) * (1 - test_size))
    x = rates[:l]

    # Scaling the data
    x = SCALER.fit_transform(x.to_numpy().reshape(-1, 1))

    x_train = x[:(int(len(x) / 5) * 4)]
    x_valid = x[(int(len(x) / 5) * 4):]
    test = rates[l:]

    # Scaling test data
    x_test = SCALER.transform(test.to_numpy().reshape(-1, 1))

    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        x_train, targets=x_train[SEQ_LENGTH:], sequence_length=SEQ_LENGTH, batch_size=BATCH_SIZE)
    valid_ds = tf.keras.utils.timeseries_dataset_from_array(
        x_valid, targets=x_valid[SEQ_LENGTH:], sequence_length=SEQ_LENGTH, batch_size=BATCH_SIZE)
    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        x_test, targets=x_test[SEQ_LENGTH:], sequence_length=SEQ_LENGTH, batch_size=BATCH_SIZE, shuffle=False)

    return train_ds, valid_ds, test, test_ds


# Function to build an RNN model with tunable hyperparameters
def build_model(hp):
    neurons_1 = hp.Int("neurons_1", min_value=8, max_value=128, step=8)
    neurons_2 = hp.Int("neurons_2", min_value=8, max_value=64, step=8)
    activation = hp.Choice("activation", ["tanh", "relu", "linear"])
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(neurons_1, activation=activation, return_sequences=True, input_shape=[None, 1]),
        tf.keras.layers.SimpleRNN(neurons_2, activation=activation),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        loss=tf.keras.losses.MeanAbsoluteError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["mae"]
    )
    return model


# Function to tune hyperparameters using KerasTuner
def tune_hyperparameters(train_ds, valid_ds, max_trials=10, executions_per_trial=1):
    tuner = kt.Hyperband(
        build_model,
        objective="val_mae",
        max_epochs=50,
        factor=3,
        directory="tuning",
        project_name="rnn_tuning"
    )

    tuner.search(train_ds, validation_data=valid_ds, epochs=100,
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps


# Function to plot actual vs predicted values
def graph(model, x_test, test_ds, distribution):
    y_pred_scaled = model.predict(test_ds)
    y_pred = SCALER.inverse_transform(y_pred_scaled)
    Y_pred = pd.Series(y_pred.flatten(), index=x_test.index[SEQ_LENGTH:])

    fig, ax = plt.subplots(figsize=(20, 7))
    plt.plot(x_test.index, x_test, label="Actual", marker=".")
    plt.plot(x_test.index[SEQ_LENGTH:], Y_pred, label="Prediction", marker="x", color="r")
    plt.legend(loc="center left")
    plt.title(f"RNN model predicting next arrival rate of a {distribution} distribution")
    plt.xlabel("Updates")
    plt.ylabel("Rate (every 120s)")
    plt.grid()
    plt.savefig(f"models/img/{distribution}.png")


# Main function to load data, tune hyperparameters, train model, and save results
def main():
    if len(sys.argv) < 2:
        print("Uso: python script.py <distribution>")
        return

    distribution = sys.argv[1].lower()
    file_path = f"models/training/{distribution}_rates.csv"
    df = load_data(file_path)
    rates = df["Rate"]

    train_ds, valid_ds, x_test, test_ds = prepare_data(rates)
    best_hps = tune_hyperparameters(train_ds, valid_ds)

    #print(f"Best hyperparameters: {best_hps.values}")
    model = build_model(best_hps)
    model.fit(train_ds, validation_data=valid_ds, epochs=100)

    graph(model, x_test, test_ds, distribution)

    with open(f"models/{distribution}_rnn.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
