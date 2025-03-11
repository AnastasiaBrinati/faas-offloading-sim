import joblib
import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv

class Model:

    def __init__ (self, name, sequence_length=7, batch_size=9):
        self.name = name
        self.epochs = 100
        self.training_rounds = 0
        self.learning_rate = 0.0001
        self.neurons = 6
        self.early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="mae", patience=10, restore_best_weights=True)
        self.opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        self.loss = tf.keras.losses.MeanAbsoluteError()

        #self.m = joblib.load(name)
        #self.m = self.setup_model()
        self.m = None

        self.training_flag = False
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


    def setup_model(self, online=False):
        if online:
            # per ora fisso
            model = tf.keras.Sequential([
                tf.keras.layers.SimpleRNN(self.neurons, activation="tanh", input_shape=[None, 1]),
                tf.keras.layers.Dense(1)  # Output layer
            ])
        else:
            model = joblib.load(self.name)
        self.m = model

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

        self.training_flag = True
        self.training_rounds = 0

    def what_prediction(self):
        # if model makes an error
        model_wins = 0
        stats_wins = 0
        minimo = min(self.sequence_length, len(self.model_predicted))
        for i in range(1, minimo):
            model_error = np.abs( self.model_predicted[-i-1] - self.actual_sequence[-i] )
            stats_error = np.abs( self.stats_predicted[-i-1] - self.actual_sequence[-i] )
            if model_error > stats_error:
                stats_wins += 1
            else:
                model_wins += 1

        return model_wins > stats_wins


    def predict (self, latest_rate, alpha, online=False, adaptive=False):

        """

        Parameters
        ----------
        latest_rate: represents the actual rate since the last update
        alpha: arrival_rate_alpha
        online: flag to signal online policy behaviour
        adaptive: flag to signal adaptive policy behaviour

        Returns
        -------
        predicted_value: encapsulated single value
                ex: input sequence <- [[0.0 0.1 0.2 0.3 0.4 0.5 0.6]]
                    predicted value <- [[0.7]]

        """

        # solo la prima volta
        if len(self.rate_sequence) < self.sequence_length:
            print(f"La policy è online: {online} e adaptive: {adaptive}")
            l = [latest_rate for i in range(self.sequence_length)]
            self.rate_sequence = l
            self.predicted_sequence.append(latest_rate)
            self.stats_predicted.append(latest_rate)
            self.model_predicted.append(latest_rate)

            self.setup_model(online)

        if online:
            # waiting for some data to be available
            if len(self.actual_sequence) > self.training_threshold:
                self.training_rounds += 1
                # training every 7 updates
                if self.training_rounds == self.sequence_length:
                    self.train()

        # salva errore rispetto all'ultimo rate predetto, il primo è filler:
        error = np.abs(latest_rate - self.predicted_sequence[-1])
        self.error_sequence.append(error)

        # start saving up arrival rates
        self.actual_sequence.append(latest_rate)
        self.rate_sequence.append(latest_rate)
        self.rate_sequence.pop(0)

        predicted_value = alpha * latest_rate + (1.0 - alpha) * self.stats_predicted[-1]
        self.stats_predicted.append(predicted_value)

        if self.training_flag:
            input_sequence = np.array(self.rate_sequence)
            input_sequence = input_sequence.reshape(1, self.sequence_length)
            model_prediction = self.m.predict(input_sequence, verbose=0)
            model_prediction = model_prediction[0][0]
            self.model_predicted.append(model_prediction)

            if adaptive:
                model_wins = self.what_prediction()
                if model_wins:
                    predicted_value = model_prediction
            else:
                predicted_value = model_prediction

        self.predicted_sequence.append(predicted_value)

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