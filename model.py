import joblib
import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv

class Model:

    def __init__ (self, name, sequence_length=7, batch_size=9):
        self.name = name
        self.epochs = 10
        self.training_rounds = 0
        self.training_flag = False
        self.learning_rate = 0.0001
        self.neurons = 6
        self.early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="mae", patience=10, restore_best_weights=True)
        self.opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        self.loss = tf.keras.losses.MeanAbsoluteError()

        #self.m = joblib.load(name)
        self.m = self.setup_model()
        self.sequence_length = sequence_length
        self.training_threshold = sequence_length
        self.batch_size = batch_size
        self.rate_sequence = []
        self.actual_sequence = []
        self.predicted_sequence = []
        self.error_sequence = []

        # predictive policy:
        self.model_predicted = []
        self.stats_predicted = []


    def setup_model(self):

        # per ora fisso
        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(self.neurons * 8, return_sequences=True, input_shape=[None, 1]),
            tf.keras.layers.SimpleRNN(self.neurons * 2, return_sequences=True),
            tf.keras.layers.SimpleRNN(self.neurons),
            tf.keras.layers.Dense(1)
        ])

        return model

    def train(self):

        training_set = self.actual_sequence
        train_ds = tf.keras.utils.timeseries_dataset_from_array(
            np.array(training_set),
            targets=training_set[self.sequence_length:],
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
        )

        self.m.compile(loss=self.loss, metrics=["mae"], optimizer=self.opt)
        history = self.m.fit(train_ds, epochs=self.epochs, callbacks=[self.early_stopping_cb], verbose=0)
        #valid_loss, valid_mae = self.m.evaluate(valid_set)
        #return valid_mae * 1e6

        self.training_rounds = 0

    def what_prediction(self):
        # if model makes an error
        model_wins = 0
        stats_wins = 0
        for i in range(1, len(self.actual_sequence)):
            model_error = np.abs( self.model_predicted[-i] - self.actual_sequence[-i] )
            stats_error = np.abs( self.stats_predicted[-i] - self.actual_sequence[-i] )
            if model_error > stats_error:
                stats_wins += 1
            else:
                model_wins += 1

        return model_wins > stats_wins


    def predict (self, latest_rate, alpha):

        if len(self.actual_sequence) > self.training_threshold:
            self.training_rounds += 1
            if self.training_rounds == self.sequence_length:
                print("rounds sufficient for training")
                self.train()
                self.training_flag = True

        """
        
        Parameters
        ----------
        latest_rate: represents the actual rate since the last update

        Returns
        -------
        predicted_value: encapsulated single value
                ex: input sequence <- [[0.0 0.1 0.2 0.3 0.4 0.5 0.6]]
                    predicted value <- [[0.7]]

        """
        # se è la prima volta che viene
        if len(self.rate_sequence) < self.sequence_length:
            l = [latest_rate for i in range(self.sequence_length)]
            self.rate_sequence = l
            self.predicted_sequence.append(latest_rate)
            self.actual_sequence.append(latest_rate)

        # salva errore rispetto all'ultimo rate predetto, il primo è filler:
        error = np.abs(latest_rate - self.predicted_sequence[-1])
        self.error_sequence.append(error)

        # start saving up arrival rates
        self.rate_sequence.append(latest_rate)
        self.rate_sequence.pop(0)

        model_wins = self.what_prediction()
        if self.training_flag and model_wins:
            input_sequence = np.array(self.rate_sequence)
            input_sequence = input_sequence.reshape(1, self.sequence_length)
            predicted_value = self.m.predict(input_sequence, verbose=0)
            predicted_value = predicted_value[0][0]
        else:
            predicted_value = alpha * latest_rate + (1.0 - alpha) * self.actual_sequence[-1]

        self.model_predicted.append(predicted_value)
        self.stats_predicted.append(alpha * latest_rate + (1.0 - alpha) * self.actual_sequence[-1])

        self.predicted_sequence.append(predicted_value)
        self.actual_sequence.append(latest_rate)

        return predicted_value

    def get_error(self):
        if len(self.error_sequence) > 0:
            # Write to CSV
            with open("results/errors/" + self.name + "_errors.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Error"])  # Header
                writer.writerows(zip(self.error_sequence))  # Combine lists into rows
            return sum(x for x in self.error_sequence)/len(self.error_sequence)
        else:
            return 0.0

    def __repr__ (self):
        return self.name

    def __lt__(self, other):
        return self.name <  other.name

    def __le__(self,other):
        return self.name <= other.name

    def __hash__ (self):
        return hash(self.name)