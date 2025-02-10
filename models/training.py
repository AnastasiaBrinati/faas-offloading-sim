import pandas as pd
import tensorflow as tf
import numpy as np
import random
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import matplotlib.pyplot as plt

BATCH_SIZE = 8
SEQ_LENGTH = 7

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Leggere il file CSV in un DataFrame
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Prepara i dati per l'addestramento del modello
def prepare_data(df, target_column):

    rates = df[target_column]
    l = int(len(rates)/500)*495
    x = rates[:l]
    x_train = x[:(int(len(x)/4)*3)]
    x_valid = x[(int(len(x)/4)*3):]
    x_test = rates[l:]

    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        x_train.to_numpy(),
        targets=x_train[SEQ_LENGTH:],
        sequence_length=SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        #shuffle=True,
        #seed=SEED
    )
    valid_ds = tf.keras.utils.timeseries_dataset_from_array(
        x_valid.to_numpy(),
        targets=x_valid[SEQ_LENGTH:],
        sequence_length=SEQ_LENGTH,
        batch_size=BATCH_SIZE
    )
    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        x_test.to_numpy(),
        targets=x_test[SEQ_LENGTH:],
        sequence_length=SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_ds, valid_ds, x_test, test_ds

def fit_and_evaluate(model, train_set, valid_set, loss=tf.keras.losses.MeanAbsoluteError(), learning_rate=0.01, epochs=10):
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=10, restore_best_weights=True)
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(loss=loss, optimizer=opt, metrics=["mae"])
    history = model.fit(train_set, validation_data=valid_set, epochs=epochs, callbacks=[early_stopping_cb])
    valid_loss, valid_mae = model.evaluate(valid_set)
    return valid_mae * 1e6

def graph(model, x_test, test_ds, distribution):
    Y_pred = model.predict(test_ds)
    Y_pred = pd.Series(Y_pred.flatten(), index=x_test.index[SEQ_LENGTH:])

    fig, ax = plt.subplots(figsize=(20, 7))
    plt.plot(x_test, label="Actual", marker=".")
    plt.plot(Y_pred, label="Prediction", marker="x", color="r")
    plt.legend(loc="center left")
    plt.title(f"RNN model predicting next arrival rate of a {distribution} distribution given sequence_length={SEQ_LENGTH}")
    plt.xlabel("tempo")
    plt.ylabel("rate")
    plt.grid()
    plt.savefig("models/img/"+distribution+".png")


if __name__ == "__main__":
    distribution = "sawtooth-wave"
    file_path = "models/training/synthetic_"+distribution+"_rates.csv"
    target_column = "Rate"

    df = load_data(file_path)
    train_ds, valid_ds, x_test, test_ds = prepare_data(df, target_column)

    """ okish model for sawtooth-wave  """
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(BATCH_SIZE * 4, activation='linear', input_shape=[None, 1], return_sequences=True),
        tf.keras.layers.SimpleRNN(BATCH_SIZE, activation='linear'),
        tf.keras.layers.Dense(1)  # no activation function by default
    ])

    mae = fit_and_evaluate(model, train_ds, valid_ds, loss=tf.keras.losses.Huber())


    """ okish model for square-wave
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(BATCH_SIZE * 2, activation='linear', input_shape=[None, 1], return_sequences=True),
        tf.keras.layers.SimpleRNN(BATCH_SIZE, activation='linear'),
        tf.keras.layers.Dense(1)  # no activation function by default
    ])

    fit_and_evaluate(model, train_ds, valid_ds, loss=tf.keras.losses.MeanAbsoluteError())
    """

    """ not-okish model for logistic-map   
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(BATCH_SIZE * 3, activation='linear', input_shape=[None, 1], return_sequences=True),
        tf.keras.layers.SimpleRNN(BATCH_SIZE * 2, activation='linear'),
        tf.keras.layers.Dense(1)  # no activation function by default
    ])

    mae = fit_and_evaluate(model, train_ds, valid_ds, loss=tf.keras.losses.MeanAbsoluteError())
    """

    """ okish model for sinusoid
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(BATCH_SIZE * 4, activation='linear', input_shape=[None, 1], return_sequences=True),
        tf.keras.layers.SimpleRNN(BATCH_SIZE * 2, activation='linear'),
        tf.keras.layers.Dense(1)  # no activation function by default
    ])
    mae = fit_and_evaluate(model, train_ds, valid_ds, loss=tf.keras.losses.Huber())
     """
    print(f"MAE: {mae}")
    graph(model, x_test, test_ds, distribution)

    with open("models/"+distribution+"_rnn.pkl", "wb") as f:
        pickle.dump(model, f)
