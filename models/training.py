import pandas as pd
import tensorflow as tf
import numpy as np
import random
from tensorflow import keras
import pickle
import matplotlib.pyplot as plt
import sys
from datasets import load_dataset

BATCH_SIZE = 9
SEQ_LENGTH = 7
FREQ = "120s"

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Leggere il file CSV in un DataFrame
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Prepara i dati per l'addestramento del modello
def prepare_data(df, test_size=0.35):
    rates = df.copy()
    l = int(len(rates)*(1-test_size))
    x = rates[:l]
    x_train = x[:(int(len(x)/5)*4)]
    x_valid = x[(int(len(x)/5)*4):]
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
    Y_pred = pd.Series(Y_pred.flatten(), index=x_test.index[SEQ_LENGTH:]) # aggiungere *120 solo per i grafici perchè almeno riportano i minuti

    fig, ax = plt.subplots(figsize=(20, 7))
    plt.plot(x_test.index, x_test, label="Actual", marker=".")      # anche qui *120
    plt.plot(Y_pred, label="Prediction", marker="x", color="r")
    plt.legend(loc="center left")
    plt.title(f"RNN model predicting next arrival rate of a {distribution} distribution given batch_size={BATCH_SIZE} and sequence_length={SEQ_LENGTH}")
    plt.xlabel("Updates")
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
        print("Uso: python script.py <sinusoid|square-wave|sawtooth-wave|logistic-map|gaussian-modulated|globus>")
        return

    distribution = sys.argv[1].lower()
    if distribution == "globus":
        data = load_dataset("anastasiafrosted/endpoint0_120", download_mode="force_redownload")
        df = pd.DataFrame(data['train'])
        # Ensure the `timestamp` column is in datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        rates = df.set_index("timestamp")
        rates = rates['avg_invocations_rate']
        # ZERO FITTING
        start_date = rates.index.min()
        end_date = df.index.max()
        time_period = pd.date_range(start=start_date, end=end_date, freq=FREQ)
        zero_fitted_train2 = df.copy()
        zero_fitted_train2 = zero_fitted_train2.reindex(time_period)
        zero_fitted_train2 = zero_fitted_train2.fillna(0)
        train_ds, valid_ds, x_test, test_ds = prepare_data(zero_fitted_train2, test_size=0.9925)
    else:
        file_path = "models/training/synthetic_"+distribution+"_rates.csv"
        df = load_data(file_path)
        rates = df["Rate"]
        print(rates)
        train_ds, valid_ds, x_test, test_ds = prepare_data(rates)

    actv = "linear"     # actv = 'tanh' is best suited for periodic functions
    loss = tf.keras.losses.MeanAbsoluteError()
    learning_rate = 0.01
    epochs=5

    if distribution == "sinusoid":
        neurons = BATCH_SIZE*4+1
        actv = "tanh"
        learning_rate = 0.0006
        epochs = 20
        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(neurons, activation=actv, input_shape=[None, 1]),
            tf.keras.layers.Dense(1)  # Output layer
        ])
    elif distribution == "shifted-sinusoid":
        neurons = BATCH_SIZE*4+1
        actv = "tanh"
        learning_rate = 0.0006
        epochs = 20
        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(neurons, activation=actv, input_shape=[None, 1]),
            tf.keras.layers.Dense(1)  # Output layer
        ])
    elif distribution == "bigger-sinusoid":
        neurons = BATCH_SIZE*7
        actv = "tanh"
        learning_rate = 0.0006
        epochs = 50
        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(neurons, activation=actv, input_shape=[None, 1]),
            tf.keras.layers.Dense(1)  # Output layer
        ])
    elif distribution == "bigger-shifted-sinusoid":
        neurons = BATCH_SIZE*7
        actv = "tanh"
        learning_rate = 0.0005
        epochs = 50
        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(neurons, activation=actv, input_shape=[None, 1]),
            tf.keras.layers.Dense(1)  # Output layer
        ])
    elif distribution == "square-wave":
        neurons_1 = BATCH_SIZE*3-2
        neurons_2 = BATCH_SIZE-2
        epochs = 50
        learning_rate = 0.009
        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(neurons_1, activation=actv, input_shape=[None, 1], return_sequences=True),
            tf.keras.layers.SimpleRNN(neurons_2, activation=actv),
            tf.keras.layers.Dense(1)  # Output layer
        ])
    elif distribution == "bigger-square-wave":
        neurons_1 = BATCH_SIZE*3-3
        neurons_2 = BATCH_SIZE*2
        epochs = 50
        actv = 'tanh'
        learning_rate = 0.01
        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(neurons_1, activation=actv, input_shape=[None, 1], return_sequences=True),
            tf.keras.layers.SimpleRNN(neurons_2, activation=actv),
            tf.keras.layers.Dense(1)  # Output layer
        ])
    elif distribution == "sawtooth-wave":
        neurons_1 = BATCH_SIZE*6
        neurons_2 = BATCH_SIZE*2-2
        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(neurons_1, activation=actv, input_shape=[None, 1], return_sequences=True),
            tf.keras.layers.SimpleRNN(neurons_2, activation=actv),
            tf.keras.layers.Dense(1)  # Output layer
        ])
    elif distribution == "logistic-map":
        neurons_1 = BATCH_SIZE*5
        neurons_2 = BATCH_SIZE*4
        loss = tf.keras.losses.Huber()
        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(neurons_1, activation=actv, input_shape=[None, 1], return_sequences=True),
            tf.keras.layers.SimpleRNN(neurons_2, activation=actv),
            tf.keras.layers.Dense(1)  # Output layer
        ])
    elif distribution == "gaussian-modulated":
        neurons_1 = BATCH_SIZE*3-2
        neurons_2 = BATCH_SIZE
        loss = tf.keras.losses.Huber()
        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(neurons_1, activation=actv, input_shape=[None, 1], return_sequences=True),
            tf.keras.layers.SimpleRNN(neurons_2, activation=actv),
            tf.keras.layers.Dense(1)  # Output layer
        ])
    elif distribution == "globus":
        neurons = 32
        loss = tf.keras.losses.Huber()
        """neurons_1 = 30
        neurons_2 = 15
        loss = tf.keras.losses.Huber()"""
        epochs=2

        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(neurons * 3, return_sequences=True, input_shape=[None, 1]),
            tf.keras.layers.SimpleRNN(neurons * 2, return_sequences=True),
            tf.keras.layers.SimpleRNN(neurons),
            tf.keras.layers.Dense(1)
        ])
    else:
        print("Distribuzione non supportata")
        return

    mae = fit_and_evaluate(model, train_ds, valid_ds, loss=loss, learning_rate=learning_rate, epochs=epochs)

    print(f"MAE: {mae}")
    graph(model, x_test, test_ds, distribution)

    with open("models/"+distribution+"_rnn.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()