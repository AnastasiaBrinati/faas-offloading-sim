from dataclasses import dataclass,field
from enum import Enum
from model import Model
import csv
import conf

class ContainerPool:

    def __init__ (self):
        self.pool = []

    def append (self, e):
        self.pool.append(e)

    def remove (self, f):
        for entry in self.pool:
            if f.name == entry[0].name:
                self.pool.remove(entry)
                return

    def __len__ (self):
        return len(self.pool)

    def front (self):
        return self.pool[0]

    def reclaim_memory (self, required_mem):
        mem = [entry[0].memory for entry in self.pool]
        if sum(mem) < required_mem:
            return 0.0
        s = sorted([e[0] for e in self.pool], reverse=True, key = lambda x: x.memory)
        reclaimed = 0
        while reclaimed < required_mem:
            f = s[0]
            s = s[1:]
            self.remove(f)
            reclaimed += f.memory
        return reclaimed

    def __contains__ (self, f):
        if not isinstance(f, Function):
            return False
        for entry in self.pool:
            if f.name == entry[0].name:
                return True
        return False
    
    def __repr__ (self):
        return repr(self.pool)

class Node:

    def __init__ (self, name, memory, speedup, region, cost=0.0,
                  custom_sched_policy=None,
                  peer_exposed_memory_fraction=1.0, model_name=None):
        self.name = name
        self.total_memory = memory
        self.curr_memory = memory
        self.memories = []
        self.peer_exposed_memory_fraction = peer_exposed_memory_fraction
        self.speedup = speedup
        self.region = region
        self.cost = cost
        self.custom_sched_policy = custom_sched_policy

        self.warm_pool = ContainerPool()
        self.kv_store = {}

        if model_name is not None:
            print(model_name)
            self.model = Model(model_name)
        else:
            self.model = None

    def predict(self, new_rate):
        return self.model.predict(new_rate)

    def get_model_error(self):
        if self.model is None:
            print("Model is: None")
            return
        # get error stats
        print(f"Node {self.name} using model {self.model}, mean absolute error: {self.model.get_error()}")

    def get_memories(self, sim):
        with open("results/memory/" + self.name + "_" + sim.config.get(conf.SEC_POLICY, conf.POLICY_NAME, fallback="basic") + ".csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Memory"])  # Header
            writer.writerows(zip(self.memories))  # Combine lists into rows
        return

    def __repr__ (self):
        return self.name

    def __lt__(self, other):
        return self.name <  other.name

    def __le__(self,other):
        return self.name <= other.name

    def __hash__ (self):
        return hash(self.name)




@dataclass
class QoSClass:
    name: str
    max_rt: float
    arrival_weight: float = 1.0
    utility: float = 1.0
    min_completion_percentage: float = 0.0
    deadline_penalty: float = 0.0
    drop_penalty: float = 0.0

    
    def __repr__ (self):
        return self.name

    def __hash__ (self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name <  other.name

    def __le__(self,other):
        return self.name <= other.name

@dataclass
class Function:
    name: str
    memory: int
    serviceMean: float
    serviceSCV: float = 1.0
    initMean: float = 0.500
    inputSizeMean: float = 100
    accessed_keys: [] = field(default_factory=lambda: [])
    max_data_access_time: float = None 
    
    def __repr__ (self):
        return self.name

    def __hash__ (self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name <  other.name

    def __le__(self,other):
        return self.name <= other.name

@dataclass
class Container:
    function: Function
    expiration_time: float

    def __eq__ (self, other):
        if not isinstance(other, Container) and not isinstance(other, Function):
            return False
        elif isinstance(other, Function):
            return self.function == other.name
        else:
            return self.function == other.function



if __name__ == "__main__":
    pool = ContainerPool()
    f = Function("a", 200, 1, 1, 1)
    f2 = Function("b", 100, 1, 1, 1)
    pool.append((f,1))
    pool.append((f2,1))
    print(pool.pool)
    pool.reclaim_memory(500)
    print(pool.pool)
    pool.reclaim_memory(10)
    print(pool.pool)
