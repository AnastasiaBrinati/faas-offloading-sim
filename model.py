import joblib
import numpy as np
import csv

class Model:

    def __init__ (self, name, sequence_length=7):
        self.name = name
        self.m = joblib.load(name)
        self.sequence_length = sequence_length
        self.rate_sequence = []
        self.error_sequence = []

        self.actual_rates = []
        self.predicted_rates = [0.0]    # otherwise actuals start one step ahead

    def predict (self, latest_rate):
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

        # salva errore rispetto all'ultimo rate predetto:
        error = latest_rate - self.rate_sequence[-1]
        self.error_sequence.append(np.abs(error))

        self.rate_sequence.append(latest_rate)
        self.rate_sequence.pop(0)

        input_sequence = np.array(self.rate_sequence)
        input_sequence = input_sequence.reshape(1, self.sequence_length)
        predicted_value = self.m.predict(input_sequence, verbose=0)

        # save stats
        self.actual_rates.append(latest_rate)
        self.predicted_rates.append(predicted_value[0][0])

        return predicted_value[0][0]

    def get_stats(self, node):
        # Write to CSV
        with open("results/predictions/" + node + "_predictions.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Actual", "Predicted"])  # Header
            writer.writerows(zip(self.actual_rates, self.predicted_rates[:-1]))  # Combine lists into rows

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