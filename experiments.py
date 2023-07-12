import sys
import os
import argparse
import pandas as pd

import faas
import conf
from arrivals import PoissonArrivalProcess, TraceArrivalProcess
from simulation import Simulation
from infrastructure import *
from main import read_spec_file

DEFAULT_CONFIG_FILE = "config.ini"
DEFAULT_OUT_DIR = "results"
DEFAULT_DURATION = 3600
SEEDS=[1,293,287844,2902,944,9573,102903,193,456,71]


def print_results (results, filename=None):
    for line in results:
        print(line)
    if filename is not None:
        with open(filename, "w") as of:
            for line in results:
                print(line,file=of)

def default_infra():
    # Regions
    reg_cloud = Region("cloud")
    reg_edge = Region("edge", reg_cloud)
    regions = [reg_edge, reg_cloud]
    # Latency
    latencies = {(reg_edge,reg_cloud): 0.100}
    bandwidth_mbps = {(reg_edge,reg_edge): 100.0, (reg_cloud,reg_cloud): 1000.0,\
            (reg_edge,reg_cloud): 10.0}
    # Infrastructure
    return Infrastructure(regions, latencies, bandwidth_mbps)

def _experiment (config):
    infra = default_infra()
    spec_file_name = config.get(conf.SEC_SIM, conf.SPEC_FILE, fallback=None)
    classes, functions, node2arrivals  = read_spec_file (spec_file_name, infra, config)
    sim = Simulation(config, infra, functions, classes, node2arrivals)
    final_stats = sim.run()
    return final_stats

def relevant_stats_dict (stats):
    result = {}
    result["Utility"] = stats.utility
    result["Penalty"] = stats.penalty
    result["NetUtility"] = stats.utility-stats.penalty
    result["Cost"] = stats.cost
    result["BudgetExcessPerc"] = max(0, (stats.cost-stats.budget)/stats.budget*100)
    return result



def experiment_main_comparison(args, config):
    results = []
    outfile=os.path.join(DEFAULT_OUT_DIR,"mainComparison.csv")

    POLICIES = ["probabilistic", "probabilistic2", "greedy", "greedy-min-cost", "greedy-budget"]

    # Check existing results
    old_results = None
    if not args.force:
        try:
            old_results = pd.read_csv(outfile)
        except:
            pass

    for seed in SEEDS:
        config.set(conf.SEC_SIM, conf.SEED, str(seed))

        for pol in POLICIES:
            config.set(conf.SEC_POLICY, conf.POLICY_NAME, pol)

            keys = {}
            keys["Policy"] = pol
            keys["Seed"] = seed

            run_string = "_".join([f"{k}{v}" for k,v in keys.items()])

            # Check if we can skip this run
            if old_results is not None and not\
                    old_results[(old_results.Seed == seed) &\
                        (old_results.Policy == pol)].empty:
                print("Skipping conf")
                continue

            stats = _experiment(config)
            with open(os.path.join(DEFAULT_OUT_DIR, f"mainComparison_{run_string}.json"), "w") as of:
                stats.print(of)

            result=dict(list(keys.items()) + list(relevant_stats_dict(stats).items()))
            results.append(result)
            print(result)
    
    resultsDf = pd.DataFrame(results)
    if old_results is not None:
        resultsDf = pd.concat([old_results, resultsDf])
    resultsDf.to_csv(outfile, index=False)
    print(resultsDf.groupby("Policy").mean())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', action='store', required=False, default="", type=str)
    parser.add_argument('--force', action='store_true', required=False, default=False)
    parser.add_argument('--debug', action='store_true', required=False, default=False)

    args = parser.parse_args()

    config = conf.parse_config_file(DEFAULT_CONFIG_FILE)
    config.set(conf.SEC_SIM, conf.STAT_PRINT_INTERVAL, "-1")
    config.set(conf.SEC_SIM, conf.CLOSE_DOOR_TIME, str(DEFAULT_DURATION))

    if args.debug:
        args.force = True
        SEEDS=SEEDS[:1]
    
    if args.experiment.lower() == "a":
        experiment_main_comparison(args, config)
    else:
        print("Unknown experiment!")
        exit(1)
