import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import pickle

SEED = 123
BATCH_SIZE = 32
SEQ_LENGTH = 15

# Leggere il file CSV in un DataFrame
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Prepara i dati per l'addestramento del modello
def prepare_data(df, target_column):

    rates = df[target_column]
    x_train = rates[(int(len(rates)/2)):]
    x_valid = rates[:int(len(rates)/2)]

    tf.random.set_seed(SEED)  # extra code – ensures reproducibility
    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        x_train.to_numpy(),
        targets=x_train[SEQ_LENGTH:],
        sequence_length=SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )
    valid_ds = tf.keras.utils.timeseries_dataset_from_array(
        x_valid.to_numpy(),
        targets=x_valid[SEQ_LENGTH:],
        sequence_length=SEQ_LENGTH,
        batch_size=BATCH_SIZE
    )

    return train_ds, valid_ds

def fit_and_evaluate(model, train_set, valid_set, learning_rate=0.01, epochs=10):
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=10, restore_best_weights=True)
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"])
    history = model.fit(train_set, validation_data=valid_set, epochs=epochs, callbacks=[early_stopping_cb])
    valid_loss, valid_mae = model.evaluate(valid_set)
    return valid_mae * 1e6


if __name__ == "__main__":
    file_path = "traces/training/synthetic_uniform_rates1.csv"  # Modifica con il percorso corretto
    target_column = "Rate"

    df = load_data(file_path)
    x_train, x_valid = prepare_data(df, target_column)

    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(BATCH_SIZE * 2, input_shape=[None, 1]),
        tf.keras.layers.Dense(1)  # no activation function by default
    ])

    fit_and_evaluate(model, x_train, x_valid)

    with open("models/uniform_rnn1.pkl", "wb") as f:
        pickle.dump(model, f)
