import pandas as pd
import tensorflow as tf
import numpy as np
import random
from tensorflow import keras
import pickle
import matplotlib.pyplot as plt
import sys

BATCH_SIZE = 9
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
    l = int(len(rates)/2)
    x = rates[:l]
    x_train = x[:(int(len(x)/4)*3)]
    x_valid = x[(int(len(x)/4)*3):]
    x_test = rates[l:]

    print(f"len(x_train): {len(x_train)}")
    print(f"len(x_test): {len(x_test)}")

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

def graph(model, x_test, test_ds, distribution):
    Y_pred = model.predict(test_ds)
    Y_pred = pd.Series(Y_pred.flatten(), index=x_test.index[SEQ_LENGTH:]*120) # ho aggiunto il *120 solo epr i grafiic perchè almeno riportano i minuti

    fig, ax = plt.subplots(figsize=(20, 7))
    plt.plot(x_test.index*120, x_test, label="Actual", marker=".")
    plt.plot(Y_pred, label="Prediction", marker="x", color="r")
    plt.legend(loc="center left")
    plt.title(f"RNN model predicting next arrival rate of a {distribution} distribution given batch_size={BATCH_SIZE} and sequence_length={SEQ_LENGTH}")
    plt.xlabel("Minutes")
    plt.ylabel("Rate (every 120s)")
    plt.grid()
    plt.savefig("models/img/"+distribution+".png")

def fit_and_evaluate(model, train_set, valid_set, loss=tf.keras.losses.MeanAbsoluteError(), learning_rate=0.01, epochs=10):
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=10, restore_best_weights=True)
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(loss=loss, metrics=["mae"], optimizer=opt)
    history = model.fit(train_set, validation_data=valid_set, epochs=epochs, callbacks=[early_stopping_cb])
    valid_loss, valid_mae = model.evaluate(valid_set)
    return valid_mae * 1e6

def main():
    if len(sys.argv) < 2:
        print("Uso: python script.py <sinusoid|square-wave|sawtooth-wave|logistic-map|gaussian-modulated>")
        return

    distribution = sys.argv[1].lower()
    file_path = "models/training/synthetic_"+distribution+"_rates.csv"
    target_column = "Rate"

    df = load_data(file_path)
    train_ds, valid_ds, x_test, test_ds = prepare_data(df, target_column)

    neurons_1 = 1
    neurons_2 = 1
    actv = "linear"
    loss = tf.keras.losses.MeanAbsoluteError()
    learning_rate = 0.01

    if distribution == "sinusoid":
        neurons_1 = BATCH_SIZE*9-1
        neurons_2 = BATCH_SIZE*2
        loss = tf.keras.losses.Huber()
    elif distribution == "square-wave":
        neurons_1 = BATCH_SIZE*3
        neurons_2 = BATCH_SIZE-3
    elif distribution == "sawtooth-wave":
        neurons_1 = BATCH_SIZE*6
        neurons_2 = BATCH_SIZE*2-2
    elif distribution == "logistic-map":
        neurons_1 = BATCH_SIZE*5
        neurons_2 = BATCH_SIZE*4
        loss = tf.keras.losses.Huber()
    elif distribution == "gaussian-modulated":
        neurons_1 = BATCH_SIZE*3-2
        neurons_2 = BATCH_SIZE
        loss = tf.keras.losses.Huber()
    else:
         print("Distribuzione non supportata!")
         return


    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(neurons_1, activation=actv, input_shape=[None, 1], return_sequences=True),
        tf.keras.layers.SimpleRNN(neurons_2, activation=actv, input_shape=[None, 1]),
        tf.keras.layers.Dense(1)  # Output layer
    ])

    mae = fit_and_evaluate(model, train_ds, valid_ds, loss=loss, learning_rate=learning_rate)

    print(f"MAE: {mae}")
    graph(model, x_test, test_ds, distribution)

    with open("models/"+distribution+"_rnn.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()