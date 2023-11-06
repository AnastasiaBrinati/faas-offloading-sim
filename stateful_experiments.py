import sys
import os
import argparse
import pandas as pd
from numpy.random import SeedSequence, default_rng

from spec import generate_temp_spec
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

def default_infra(edge_cloud_latency=0.100):
    # Regions
    reg_cloud = Region("cloud")
    reg_edge = Region("edge", reg_cloud)
    regions = [reg_edge, reg_cloud]
    # Latency
    latencies = {(reg_edge,reg_cloud): edge_cloud_latency, (reg_edge,reg_edge): 0.005}
    bandwidth_mbps = {(reg_edge,reg_edge): 100.0, (reg_cloud,reg_cloud): 1000.0,\
            (reg_edge,reg_cloud): 10.0}
    # Infrastructure
    return Infrastructure(regions, latencies, bandwidth_mbps)

def _experiment (config, infra, spec_file_name):
    seed = config.getint(conf.SEC_SIM, conf.SEED, fallback=1)
    seed_sequence = SeedSequence(seed)

    classes, functions, node2arrivals  = read_spec_file (spec_file_name, infra, config)
    sim = Simulation(config, seed_sequence, infra, functions, classes, node2arrivals)
    final_stats = sim.run()
    del(sim)
    return final_stats

def relevant_stats_dict (stats):
    result = {}
    result["Utility"] = stats.utility
    result["Penalty"] = stats.penalty
    result["NetUtility"] = stats.utility-stats.penalty
    result["Cost"] = stats.cost
    result["BudgetExcessPerc"] = max(0, (stats.cost-stats.budget)/stats.budget*100)
    return result


def experiment_simple (args, config):
    results = []
    exp_tag = "simple"
    outfile=os.path.join(DEFAULT_OUT_DIR,f"{exp_tag}.csv")

    config.set(conf.SEC_POLICY, conf.CLOUD_COLD_START_EST_STRATEGY, "pacs")
    config.set(conf.SEC_POLICY, conf.EDGE_COLD_START_EST_STRATEGY, "pacs")
    config.set(conf.SEC_POLICY, conf.POLICY_UPDATE_INTERVAL, "120")
    config.set(conf.SEC_POLICY, conf.POLICY_ARRIVAL_RATE_ALPHA, "0.3")


    POLICIES = ["basic", "basic-edge", "basic-budget", "probabilistic2", "greedy-budget",  "probabilistic2-strict",
                "probabilistic2Alt", "probabilistic2-strictAlt"]

    # Check existing results
    old_results = None
    if not args.force:
        try:
            old_results = pd.read_csv(outfile)
        except:
            pass

    config.set(conf.SEC_POLICY, conf.HOURLY_BUDGET, "1")

    for seed in SEEDS:
        config.set(conf.SEC_SIM, conf.SEED, str(seed))
        for cloud_speedup in [1.0, 2.0, 4.0]:
            for cloud_cost in [0.00001, 0.0001, 0.001]:
                for load_coeff in [0.5, 1, 2, 4]:
                    for pol in POLICIES:
                        config.set(conf.SEC_POLICY, conf.POLICY_NAME, pol)

                        if "greedy" in pol:
                            config.set(conf.SEC_POLICY, conf.LOCAL_COLD_START_EST_STRATEGY, "full-knowledge")
                        else:
                            config.set(conf.SEC_POLICY, conf.LOCAL_COLD_START_EST_STRATEGY, "naive-per-function")


                        keys = {}
                        keys["Policy"] = pol
                        keys["Seed"] = seed
                        keys["CloudCost"] = cloud_cost
                        keys["CloudSpeedup"] = cloud_speedup
                        keys["Load"] = load_coeff

                        run_string = "_".join([f"{k}{v}" for k,v in keys.items()])

                        # Check if we can skip this run
                        if old_results is not None and not\
                                old_results[(old_results.Seed == seed) &\
                                    (old_results.CloudSpeedup == cloud_speedup) &\
                                    (old_results.CloudCost == cloud_cost) &\
                                    (old_results.Load == load_coeff) &\
                                    (old_results.Policy == pol)].empty:
                            print("Skipping conf")
                            continue

                        temp_spec_file = generate_temp_spec (load_coeff=load_coeff, cloud_cost=cloud_cost, cloud_speedup=cloud_speedup)
                        infra = default_infra()
                        stats = _experiment(config, infra, temp_spec_file.name)
                        temp_spec_file.close()
                        with open(os.path.join(DEFAULT_OUT_DIR, f"{exp_tag}_{run_string}.json"), "w") as of:
                            stats.print(of)

                        result=dict(list(keys.items()) + list(relevant_stats_dict(stats).items()))
                        results.append(result)
                        print(result)

                        resultsDf = pd.DataFrame(results)
                        if old_results is not None:
                            resultsDf = pd.concat([old_results, resultsDf])
                        resultsDf.to_csv(outfile, index=False)
    
    resultsDf = pd.DataFrame(results)
    if old_results is not None:
        resultsDf = pd.concat([old_results, resultsDf])
    resultsDf.to_csv(outfile, index=False)
    print(resultsDf.groupby("Policy").mean())

    with open(os.path.join(DEFAULT_OUT_DIR, f"{exp_tag}_conf.ini"), "w") as of:
        config.write(of)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', action='store', required=False, default="", type=str)
    parser.add_argument('--force', action='store_true', required=False, default=False)
    parser.add_argument('--debug', action='store_true', required=False, default=False)
    parser.add_argument('--seed', action='store', required=False, default=None, type=int)

    args = parser.parse_args()

    config = conf.parse_config_file("default.ini")
    config.set(conf.SEC_SIM, conf.STAT_PRINT_INTERVAL, "-1")
    config.set(conf.SEC_SIM, conf.CLOSE_DOOR_TIME, str(DEFAULT_DURATION))

    if args.debug:
        args.force = True
        SEEDS=SEEDS[:1]

    if args.seed is not None:
        SEEDS = [int(args.seed)]
    
    if args.experiment.lower() == "a":
        experiment_simple(args, config)
    else:
        print("Unknown experiment!")
        exit(1)
