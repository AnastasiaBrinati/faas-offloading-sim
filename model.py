import joblib
import numpy as np

class Model:

    def __init__ (self, name, sequence_length=7):
        self.name = name
        self.m = joblib.load(name)
        self.sequence_length = sequence_length
        self.rate_sequence = []
        self.error_sequence = []

    def predict (self, new_rate):
        # se è la prima volta che viene
        if len(self.rate_sequence) < self.sequence_length:
            l = [new_rate for i in range(self.sequence_length)]
            self.rate_sequence = l

        # salva errore rispetto all'ultimo rate predetto:
        error = new_rate - self.rate_sequence[-1]
        self.error_sequence.append(np.abs(error))

        self.rate_sequence.append(new_rate)
        self.rate_sequence.pop(0)
        #print(f"self.rate_sequence: {self.rate_sequence}")

        input_sequence = np.array(self.rate_sequence)
        input_sequence = input_sequence.reshape(1, self.sequence_length)
        predicted_value = self.m.predict(input_sequence)
        return predicted_value[0][0]

    def __repr__ (self):
        return self.name

    def __lt__(self, other):
        return self.name <  other.name

    def __le__(self,other):
        return self.name <= other.name

    def __hash__ (self):
        return hash(self.name)