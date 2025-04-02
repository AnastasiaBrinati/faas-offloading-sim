import yaml
import tempfile
from scipy.stats import truncnorm
import numpy as np
from numpy.random import uniform, choice
import map as MAP
import random

ARRIVAL_DISTRIBUTIONS = ["exp","mmpp","erlang","hyper"]

# Returns an open NamedTemporaryFile
def generate_temp_spec (n_functions=5, load_coeff=1.0, dynamic_rate_coeff=1.0, arrivals_to_single_node=True,
                   n_classes=4, cloud_cost=0.00005, cloud_speedup=1.0, n_edges=5):

    classes = [{'name': 'critical', 'max_resp_time': 0.5, 'utility': 1.0, 'arrival_weight': 1.0}, {'name': 'standard', 'max_resp_time': 0.5, 'utility': 0.01, 'arrival_weight': 7.0}, {'name': 'batch', 'max_resp_time': 99.0, 'utility': 1.0, 'arrival_weight': 1.0}, {'name': 'criticalP', 'max_resp_time': 0.5, 'utility': 1.0, 'arrival_weight': 1.0}]
    nodes = [{'name': 'edge1', 'region': 'edge', 'memory': 4096}, {'name': 'edge2', 'region': 'edge', 'memory': 4096}, {'name': 'edge3', 'region': 'edge', 'memory': 4096}, {'name': 'edge4', 'region': 'edge', 'memory': 4096}, {'name': 'edge5', 'region': 'edge', 'memory': 4096}, {'name': 'cloud1', 'region': 'cloud', 'cost': cloud_cost, 'speedup': cloud_speedup, 'memory': 128000}]
    functions = [{'name': 'f1', 'memory': 512, 'duration_mean': 0.4, 'duration_scv': 1.0, 'init_mean': 0.5}, {'name': 'f2', 'memory': 512, 'duration_mean': 0.2, 'duration_scv': 1.0, 'init_mean': 0.25}, {'name': 'f3', 'memory': 128, 'duration_mean': 0.3, 'duration_scv': 1.0, 'init_mean': 0.6}, {'name': 'f4', 'memory': 1024, 'duration_mean': 0.25, 'duration_scv': 1.0, 'init_mean': 0.25}, {'name': 'f5', 'memory': 256, 'duration_mean': 0.45, 'duration_scv': 1.0, 'init_mean': 0.5}]
   
    #Extend functions list if needed
    if n_functions > len(functions):
        i=0
        while n_functions > len(functions):
            new_f = functions[i].copy()
            new_f["name"] = f"f{len(functions)+1}"
            functions.append(new_f)
            i+=1
    else:
        functions = functions[:n_functions]
    function_names = [f["name"] for f in functions]

    #Extend node list if needed
    if n_edges > len(nodes) - 1:
        i=0
        while n_edges > len(nodes) - 1:
            new_f = nodes[0].copy()
            new_f["name"] = f"nedge{i}"
            nodes.append(new_f)
            i+=1
    elif n_edges < len(nodes) - 1:
        new_nodes = nodes[:n_edges]
        new_nodes.append(nodes[-1])
        nodes = new_nodes

    #Extend class list if needed
    if n_classes > len(classes):
        i=0
        while n_classes > len(classes):
            new_f = classes[i].copy()
            new_f["name"] = f"c{len(classes)+1}"
            classes.append(new_f)
            i+=1
    else:
        classes = classes[:n_classes]

    ntemp = tempfile.NamedTemporaryFile(mode="w")
    write_spec(ntemp, functions, classes, nodes, load_coeff, dynamic_rate_coeff, arrivals_to_single_node)
    return ntemp

def generate_random_temp_spec (rng, n_functions=6, load_coeff=1.0, dynamic_rate_coeff=1.0, arrivals_to_single_node=True,
                   n_classes=3, cloud_cost=0.0001, cloud_speedup=1.0, n_edges=2):

    classes = [{'name': 'critical', 'max_resp_time': 0.5, 'utility': 1.0, 'arrival_weight': 1.0},
               {'name': 'best-effort', 'max_resp_time': 0.5, 'utility': 0.1, 'arrival_weight': 1.0},
               {'name': 'deferrable', 'max_resp_time': 5.0, 'utility': 0.5, 'arrival_weight': 1.0},
               {'name': 'standard', 'max_resp_time': 0.5, 'utility': 0.01, 'arrival_weight': 7.0},
               {'name': 'batch', 'max_resp_time': 99.0, 'utility': 1.0, 'arrival_weight': 1.0},
               {'name': 'criticalP', 'max_resp_time': 0.5, 'utility': 1.0, 'arrival_weight': 1.0}]
    nodes = [{'name': 'edge1', 'region': 'edge', 'memory': 2048},
             {'name': 'edge2', 'region': 'edge', 'memory': 2048},
             {'name': 'cloud1', 'region': 'cloud', 'cost': cloud_cost, 'speedup': cloud_speedup, 'memory': 32000}]
    functions = [{'name': 'f1', 'memory': 512, 'duration_mean': 0.4, 'duration_scv': 1.0, 'init_mean': 0.5, 'edge': 'edge1', 'trace': "traces/synthetic/debs15_1_interarrivals.csv", 'model': "models/debs15_1_rnn.pkl"},
                 {'name': 'f2', 'memory': 512, 'duration_mean': 0.4, 'duration_scv': 1.0, 'init_mean': 0.25, 'edge': 'edge1', 'trace': "traces/synthetic/debs15_2_interarrivals.csv", 'model': "models/debs15_1_rnn.pkl"},
                 {'name': 'f3', 'memory': 128, 'duration_mean': 0.2, 'duration_scv': 1.0, 'init_mean': 0.6, 'edge': 'edge1', 'trace': "traces/synthetic/globus_arrivals.csv", 'model': "models/globus_rnn.pkl"},
                 {'name': 'f4', 'memory': 128, 'duration_mean': 0.2, 'duration_scv': 1.0, 'init_mean': 0.25, 'edge': 'edge2', 'trace': "traces/synthetic/shifted-sinusoid_arrivals.csv", 'model': "models/shifted-sinusoid_rnn.pkl"},
                 {'name': 'f5', 'memory': 128, 'duration_mean': 0.2, 'duration_scv': 1.0, 'init_mean': 0.5, 'edge': 'edge2', 'trace': "traces/synthetic/square-wave_arrivals.csv", 'model': "models/square-wave_rnn.pkl"},
                 {'name': 'f6', 'memory': 128, 'duration_mean': 0.2, 'duration_scv': 1.0, 'init_mean': 0.6, 'edge': 'edge2', 'trace': "traces/synthetic/sawtooth-wave_arrivals.csv", 'model': "models/sawtooth-wave_rnn.pkl"}]
   
    #Extend functions list if needed
    if n_functions > len(functions):
        i=0
        while n_functions > len(functions):
            new_f = functions[i].copy()
            new_f["name"] = f"f{len(functions)+1}"
            functions.append(new_f)
            i+=1
    else:
        functions = functions[:n_functions]
    function_names = [f["name"] for f in functions]

    #Extend node list if needed
    if n_edges > len(nodes) - 1:
        i=0
        while n_edges > len(nodes) - 1:
            new_f = nodes[0].copy()
            new_f["name"] = f"nedge{i}"
            nodes.append(new_f)
            i+=1
    elif n_edges < len(nodes) - 1:
        new_nodes = nodes[:n_edges]
        new_nodes.append(nodes[-1])
        nodes = new_nodes

    #Extend class list if needed
    if n_classes > len(classes):
        i=0
        while n_classes > len(classes):
            new_f = classes[i].copy()
            new_f["name"] = f"c{len(classes)+1}"
            classes.append(new_f)
            i+=1
    else:
        classes = classes[:n_classes]

    #print(f"funcs: {functions}")
    #print(f"classes: {classes}")
    #print(f"nodes: {nodes}")

    # Randomly set specs
    for f in functions:
        #f["duration_mean"] = float(rng.uniform(0.3,0.4))
        f["init_mean"] = float(rng.uniform(0.25,0.75))
        #f["memory"] = float(rng.uniform(128,512))
        #f["duration_scv"] = float(rng.choice([1,0.5,0.25]))

        myclip_a = 100
        myclip_b = 1024*1024*5
        loc = 1024
        scale = 10**3
        a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
        f["input_mean"] = float(truncnorm.rvs(a, b, loc=loc, scale=scale, size=1))

    if dynamic_rate_coeff == 0.0:
        print("Setting distribution to: trace")
        functions_arrival_distributions = {f["name"]: "trace" for f in functions}
    elif dynamic_rate_coeff > 1.0:
        print("Forcing Poisson arrivals")
        functions_arrival_distributions = {f["name"]: "exp" for f in functions}
    else:
        functions_arrival_distributions = rng.choice(ARRIVAL_DISTRIBUTIONS, size=n_functions)
        print(f"functions_arrival_distributions:{functions_arrival_distributions}")
        functions_arrival_distributions = {f["name"]: d for f,d in zip(functions, functions_arrival_distributions)}


    ntemp = tempfile.NamedTemporaryFile(mode="w")
    write_spec(ntemp, functions, classes, nodes, load_coeff, dynamic_rate_coeff, arrivals_to_single_node, functions_arrival_distributions=functions_arrival_distributions)
    return ntemp

# Writes a spec file to outf 
def write_spec (outf, functions, classes, nodes, load_coeff=1.0, dynamic_rate_coeff=1.0, arrivals_to_single_node=True, functions_arrival_distributions=None):

    total_fun_weight = sum([f["duration_mean"]*f["memory"] for f in functions])

    def make_arrival_dict (_node, _func, _dynamic_coeff=None, _rate=None, _map=None, _trace=None, _model=None):
        if _trace is not None:
            return {"node": _node, "function": _func, "trace": _trace, "model": _model}
        elif _rate is not None:
            return {"node": _node, "function": _func, "rate": _rate, "dynamic_coeff": _dynamic_coeff }
        elif _map is not None:
            D0,D1=_map
            mapstr=""
            for x in np.nditer(D0):
                mapstr = mapstr + f"{x};"
            for x in np.nditer(D1):
                mapstr = mapstr + f"{x};"
            mapstr=mapstr[:-1]

            arrival_dictionary = {"node": _node, "function": _func, "map": mapstr, "dynamic_coeff": _dynamic_coeff }
            return arrival_dictionary

    if functions_arrival_distributions is None:
        functions_arrival_distributions={}

    arrivals = []
    if arrivals_to_single_node:
        print("SINGLE NODE")
        rate = 10*load_coeff
        for f in functions:
            # "exp" is the default value in case f['name'] does not exist
            distr = functions_arrival_distributions.get(f["name"], "exp")
            if distr == "trace":
                arv = make_arrival_dict("edge1", f["name"], _trace=f["trace"], _model=f["model"])
            elif distr == "exp":
                arv = make_arrival_dict("edge1", f["name"], dynamic_rate_coeff, _rate=rate)
            else:
                if distr == "erlang":
                    D0,D1 = MAP.make_erlang2(rate)
                elif distr == "hyper":
                    D0,D1 = MAP.make_hyper(rate)
                elif distr == "mmpp":
                    D0,D1 = MAP.make_mmpp2(rate)
                else:
                    raise ValueError(f"unknow distr: {distr}")
                arv = make_arrival_dict("edge1", f["name"], dynamic_rate_coeff, _map=(D0,D1))
            arrivals.append(arv)

    else:
        print("ARRIVALS ON MULTIPLE NODES")
        edge_nodes = [n for n in nodes if "edge" in n["name"]]
        rate = 10*load_coeff/len(edge_nodes)
        for f in functions:
            distr = functions_arrival_distributions.get(f["name"], "exp")
            if distr == "trace":
                arv = make_arrival_dict(f["edge"], f["name"], _trace=f["trace"], _model=f["model"])
            elif distr == "exp":
                arv = make_arrival_dict(f["edge"], f["name"], dynamic_rate_coeff, _rate=rate)
            else:
                if distr == "erlang":
                    D0,D1 = MAP.make_erlang2(rate)
                elif distr == "hyper":
                    D0,D1 = MAP.make_hyper(rate)
                elif distr == "mmpp":
                    D0,D1 = MAP.make_mmpp2(rate)
                else:
                    raise ValueError(f"unknow distr: {distr}")
                arv = make_arrival_dict(f["edge"], f["name"], dynamic_rate_coeff, _map=(D0,D1))
            arrivals.append(arv)

    spec = {'classes': classes, 'nodes': nodes, 'functions': functions, 'arrivals': arrivals}
    outf.write(yaml.dump(spec))
    #print(spec)
    outf.flush()

if __name__ == "__main__":
    with open("spec.yml", "w") as outf:
        write_spec(outf, load_coeff=4, cloud_cost=0.001)
