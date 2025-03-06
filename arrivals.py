from faas import Function, QoSClass
from model import Model
from map import SamplesFromMAP, MapMean


class ArrivalProcess:

    def __init__ (self, function: Function, classes: [QoSClass]):
        self.function = function
        self.classes = classes

        self.class_rng = None
        self.iat_rng = None

        total_weight = sum([c.arrival_weight for c in classes])
        self.class_probs = [c.arrival_weight/total_weight for c in classes]

    def init_rng (self, class_rng, iat_rng, rate_rng=None):
        self.class_rng = class_rng
        self.iat_rng = iat_rng
        self.rate_rng = rate_rng

    def next_iat (self):
        raise RuntimeError("Not implemented")

    def next_class (self):
        return self.class_rng.choice(self.classes, p=self.class_probs)

    def close(self):
        pass

    def has_dynamic_rate (self):
        return False

    def get_mean_rate(self):
        raise RuntimeError("Not implemented")

    def get_per_class_mean_rate (self):
        d={}
        for c,p in zip(self.classes, self.class_probs):
            d[c] = self.get_mean_rate()*p
        return d

class PoissonArrivalProcess (ArrivalProcess):

    def __init__ (self, function: Function, classes: [QoSClass], rate: float,
                  dynamic_rate_coeff: float = 0.0):
        super().__init__(function, classes) 
        self.rate = rate
        self.dynamic_rate_coeff = dynamic_rate_coeff

        if self.has_dynamic_rate():
            self._min_rate = rate/self.dynamic_rate_coeff
            self._max_rate = rate*self.dynamic_rate_coeff

    def next_iat (self):
        return self.iat_rng.exponential(1.0/self.rate)

    def has_dynamic_rate (self):
        return self.dynamic_rate_coeff > 1.0

    def update_dynamic_rate (self):
        if self.has_dynamic_rate():
            next_min = max(self._min_rate, self.rate/self.dynamic_rate_coeff)
            next_max = min(self._max_rate, self.rate*self.dynamic_rate_coeff)
            next_rate = self.rate_rng.uniform(next_min, next_max)
            print(f"Rate: {self.rate} -> {next_rate}")
            self.rate = next_rate

    def get_mean_rate(self):
        return self.rate

class DeterministicArrivalProcess (PoissonArrivalProcess):

    def __init__ (self, function: Function, classes: [QoSClass], rate: float,
                  dynamic_rate_coeff: float = 0.0):
        super().__init__(function, classes, rate, dynamic_rate_coeff) 

    def next_iat (self):
        return 1.0/self.rate

    def get_mean_rate(self):
        return self.rate

class MAPArrivalProcess (ArrivalProcess):

    def __init__ (self, function: Function, classes: [QoSClass], D0, D1):
        super().__init__(function, classes) 
        self.state = 0
        self.D0 = D0
        self.D1 = D1
        print(f"Mean rate: {self.get_mean_rate()}")

    def next_iat (self):
        iat, self.state = SamplesFromMAP(self.D0, self.D1, 1, initial=self.state)
        return float(iat)

    def has_dynamic_rate (self):
        return False

    def get_mean_rate(self):
        return 1.0/MapMean(self.D0, self.D1)

class TraceArrivalProcess (ArrivalProcess):

    def __init__ (self, function: Function, classes: [QoSClass], trace: str, model: Model):
        super().__init__(function, classes)
        self.trace = open(trace, "r")
        if model is not None:
            self.model = Model(model)
        else:
            self.model = None

    def next_iat (self):
        try:
            return float(self.trace.readline().strip())
        except:
            return -1.0

    def predict(self, new_rate, alpha):
        return self.model.predict(new_rate, alpha)

    def get_model_error(self):
        if self.model is None:
            print("Model is: None")
            return
        # get error stats
        print(f"Function {self.function} using model {self.model}, mean absolute error: {self.model.get_error()}")

    def close(self):
        super().close()
        self.trace.close()
        self.get_model_error()
