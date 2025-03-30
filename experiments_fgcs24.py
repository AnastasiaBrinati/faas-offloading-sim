import sys
import tempfile
import os
import argparse
import pandas as pd
import numpy as np

from spec import generate_temp_spec, generate_random_temp_spec
import faas
import conf
from arrivals import PoissonArrivalProcess, TraceArrivalProcess, MAPArrivalProcess
from simulation import Simulation
from infrastructure import *
from main import read_spec_file
from numpy.random import SeedSequence, default_rng

DEFAULT_CONFIG_FILE = "config.ini"
DEFAULT_OUT_DIR = "results/results/"
DEFAULT_DURATION = 3600
SEEDS=[1,293,287844,2902,944,9573,102903,193,456,71]
PERCENTILES=np.array([1,5,10,25,50,75,90,95,99])/100.0


def print_results (results, filename=None):
    #for line in results:
    #    print(line)
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

#def _experiment (config, infra, spec_file_name):
#    classes, functions, node2arrivals  = read_spec_file (spec_file_name, infra, config)
#    sim = Simulation(config, infra, functions, classes, node2arrivals)
#    final_stats = sim.run()
#    del(sim)
#    return final_stats
def _experiment (config, seed_sequence, infra, spec_file_name, return_resp_times_stats=False):
    classes, functions, node2arrivals  = read_spec_file (spec_file_name, infra, config)
    #generate_latencies(infra, default_rng(seed_sequence.spawn(1)[0]))

    with tempfile.NamedTemporaryFile() as rtf:
        if return_resp_times_stats:
            config.set(conf.SEC_SIM, conf.RESP_TIMES_FILE, rtf.name)
        sim = Simulation(config, seed_sequence, infra, functions, classes, node2arrivals)
        final_stats = sim.run()
        del(sim)

        if return_resp_times_stats:
            # Retrieve response times
            df = pd.read_csv(rtf.name)
            # Compute percentiles
            rt_p = {f"RT-{k}": v for k,v in df.RT.quantile(PERCENTILES).items()}
            dat_p = {f"DAT-{k}": v for k,v in df.DataAccess.quantile(PERCENTILES).items()}

    if return_resp_times_stats:
        return final_stats, df, rt_p, dat_p
    else:
        return final_stats

def relevant_stats_dict (stats):
    result = {}
    result["Utility"] = stats.utility
    #result["Penalty"] = stats.penalty
    #result["NetUtility"] = stats.utility-stats.penalty
    result["Cost"] = stats.cost
    #result["BudgetExcessPerc"] = max(0, (stats.cost-stats.budget)/stats.budget*100)
    return result


def experiment_varying_arrivals (args, config):
    results = []
    exp_tag = "varyingArrivalsNew"
    outfile=os.path.join(DEFAULT_OUT_DIR,f"{exp_tag}.csv")

    config.set(conf.SEC_POLICY, conf.CLOUD_COLD_START_EST_STRATEGY, "pacs")
    config.set(conf.SEC_POLICY, conf.EDGE_COLD_START_EST_STRATEGY, "pacs")
    config.set(conf.SEC_SIM, conf.RATE_UPDATE_INTERVAL, str(60))



    POLICIES = ["basic-budget", "probabilistic", "greedy-budget", "probabilistic-strict", "probabilistic-strictAlt"]

    # Check existing results
    old_results = None
    if not args.force:
        try:
            old_results = pd.read_csv(outfile)
        except:
            pass

    for dyn_rate_coeff in [5,10,2]:
        for seed in SEEDS:
            seed_sequence = SeedSequence(seed)
            config.set(conf.SEC_SIM, conf.SEED, str(seed))
            for budget in [1,5,10]:
                config.set(conf.SEC_POLICY, conf.HOURLY_BUDGET, str(budget))
                for policy_update_interval in [60, 120]:
                    config.set(conf.SEC_POLICY, conf.POLICY_UPDATE_INTERVAL, str(policy_update_interval))
                    for alpha in [0.3, 0.5, 1.0]: 
                        config.set(conf.SEC_POLICY, conf.POLICY_ARRIVAL_RATE_ALPHA, str(alpha))
                        for pol in POLICIES:
                            config.set(conf.SEC_POLICY, conf.POLICY_NAME, pol)

                            if alpha > 0.3 and (not "probabilistic-legacy" in pol):
                                continue
                            if policy_update_interval != 120 and (not "probabilistic-legacy" in pol):
                                continue

                            if "greedy" in pol:
                                config.set(conf.SEC_POLICY, conf.LOCAL_COLD_START_EST_STRATEGY, "full-knowledge")
                            else:
                                config.set(conf.SEC_POLICY, conf.LOCAL_COLD_START_EST_STRATEGY, "naive-per-function")


                            keys = {}
                            keys["Policy"] = pol
                            keys["Seed"] = seed
                            keys["PolicyUpdInterval"] = policy_update_interval
                            keys["Alpha"] = alpha
                            keys["Budget"] = budget
                            keys["DynCoeff"] = dyn_rate_coeff

                            run_string = "_".join([f"{k}{v}" for k,v in keys.items()])

                            # Check if we can skip this run
                            if old_results is not None and not\
                                    old_results[(old_results.Seed == seed) &\
                                        (old_results.Alpha == alpha) &\
                                        (old_results.Budget == budget) &\
                                        (old_results.PolicyUpdInterval == policy_update_interval) &\
                                        (old_results.DynCoeff == dyn_rate_coeff) &\
                                        (old_results.Policy == pol)].empty:
                                print("Skipping conf")
                                continue

                            rng = default_rng(seed_sequence.spawn(1)[0])
                            temp_spec_file = generate_random_temp_spec (rng, dynamic_rate_coeff=dyn_rate_coeff)
                            infra = default_infra()
                            stats = _experiment(config, seed_sequence, infra, temp_spec_file.name)

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

    temp_spec_file.close()

    with open(os.path.join(DEFAULT_OUT_DIR, f"{exp_tag}_conf.ini"), "w") as of:
        config.write(of)


def experiment_main_comparison(args, config):
    results = []
    exp_tag = "mainComparison"
    config.set(conf.SEC_POLICY, conf.SPLIT_BUDGET_AMONG_EDGE_NODES, "false")
    outfile=os.path.join(DEFAULT_OUT_DIR,f"{exp_tag}.csv")

    config.set(conf.SEC_POLICY, conf.CLOUD_COLD_START_EST_STRATEGY, "pacs")
    config.set(conf.SEC_POLICY, conf.EDGE_COLD_START_EST_STRATEGY, "pacs")
    config.set(conf.SEC_POLICY, conf.POLICY_UPDATE_INTERVAL, "120")
    config.set(conf.SEC_POLICY, conf.POLICY_ARRIVAL_RATE_ALPHA, "0.3")


    #POLICIES = ["random", "basic", "basic-edge", "basic-budget", "probabilistic-legacy", "probabilistic", "greedy", "greedy-min-cost", "greedy-budget", "probabilistic-legacy-strict", "probabilistic-strict", "probabilistic-strictAlt", "probabilisticAlt"]
    POLICIES = ["probabilistic-function", "predictive-function",\
                "online-predictive-function", "online-adaptive-function",\
                "probabilistic-memory-function", "predictive-memory-function",\
                "online-predictive-memory-function", "online-adaptive-memory-function"]

    # Check existing results
    old_results = None
    if not args.force:
        try:
            old_results = pd.read_csv(outfile)
        except:
            pass

    for seed in SEEDS:
        config.set(conf.SEC_SIM, conf.SEED, str(seed))
        seed_sequence = SeedSequence(seed)
        for latency in [0.050, 0.100, 0.200]:
            for budget in [1,2,10,20]:
                config.set(conf.SEC_POLICY, conf.HOURLY_BUDGET, str(budget))
                for functions in [3]:
                    for pol in POLICIES:
                        print(f"policy: {pol}")
                        config.set(conf.SEC_POLICY, conf.POLICY_NAME, pol)

                        if "greedy" in pol:
                            config.set(conf.SEC_POLICY, conf.LOCAL_COLD_START_EST_STRATEGY, "full-knowledge")
                        else:
                            config.set(conf.SEC_POLICY, conf.LOCAL_COLD_START_EST_STRATEGY, "naive-per-function")

                        keys = {}
                        keys["Policy"] = pol
                        keys["Seed"] = seed
                        keys["Functions"] = functions
                        keys["Latency"] = latency
                        keys["Budget"] = budget

                        run_string = "_".join([f"{k}{v}" for k,v in keys.items()])

                        # Check if we can skip this run
                        if old_results is not None and not\
                                old_results[(old_results.Seed == seed) &\
                                    (old_results.Latency == latency) &\
                                    (old_results.Functions == functions) &\
                                    (old_results.Budget == budget) &\
                                    (old_results.Policy == pol)].empty:
                            print("Skipping conf")
                            continue

                        rng = default_rng(seed_sequence.spawn(1)[0])
                        temp_spec_file = generate_random_temp_spec (rng, n_functions=functions, dynamic_rate_coeff=0.0)
                        infra = default_infra(edge_cloud_latency=latency)
                        stats = _experiment(config, seed_sequence, infra, temp_spec_file.name)
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


def experiment_arrivals_to_all (args, config):
    results = []
    exp_tag = "mainComparisonArvAll"
    config.set(conf.SEC_POLICY, conf.SPLIT_BUDGET_AMONG_EDGE_NODES, "true")
    outfile=os.path.join(DEFAULT_OUT_DIR,f"{exp_tag}.csv")

    config.set(conf.SEC_POLICY, conf.CLOUD_COLD_START_EST_STRATEGY, "pacs")
    config.set(conf.SEC_POLICY, conf.EDGE_COLD_START_EST_STRATEGY, "pacs")
    config.set(conf.SEC_POLICY, conf.POLICY_UPDATE_INTERVAL, "120")
    config.set(conf.SEC_POLICY, conf.POLICY_ARRIVAL_RATE_ALPHA, "0.3")


    POLICIES = ["basic", "basic-budget", "probabilistic-legacy", "probabilistic", "greedy", "greedy-budget", "probabilistic-legacy-strict",
                "probabilistic-strict", "probabilistic-strictAlt", "probabilisticAlt"]

    # Check existing results
    old_results = None
    if not args.force:
        try:
            old_results = pd.read_csv(outfile)
        except:
            pass

    for seed in SEEDS:
        seed_sequence = SeedSequence(seed)
        config.set(conf.SEC_SIM, conf.SEED, str(seed))
        for latency in [0.100]:
            for budget in [0.5,1,2]:
                config.set(conf.SEC_POLICY, conf.HOURLY_BUDGET, str(budget))
                for pol in POLICIES:
                    config.set(conf.SEC_POLICY, conf.POLICY_NAME, pol)

                    if "greedy" in pol:
                        config.set(conf.SEC_POLICY, conf.LOCAL_COLD_START_EST_STRATEGY, "full-knowledge")
                    else:
                        config.set(conf.SEC_POLICY, conf.LOCAL_COLD_START_EST_STRATEGY, "naive-per-function")


                    keys = {}
                    keys["Policy"] = pol
                    keys["Seed"] = seed
                    keys["Latency"] = latency
                    keys["Budget"] = budget

                    run_string = "_".join([f"{k}{v}" for k,v in keys.items()])

                    # Check if we can skip this run
                    if old_results is not None and not\
                            old_results[(old_results.Seed == seed) &\
                                (old_results.Latency == latency) &\
                                (old_results.Budget == budget) &\
                                (old_results.Policy == pol)].empty:
                        print("Skipping conf")
                        continue

                    rng = default_rng(seed_sequence.spawn(1)[0])
                    temp_spec_file = generate_random_temp_spec (rng, n_functions=5, arrivals_to_single_node=False)
                    infra = default_infra(edge_cloud_latency=latency)
                    stats = _experiment(config, seed_sequence, infra, temp_spec_file.name)
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

def experiment_edge (args, config):
    results = []
    exp_tag = "edge"
    outfile=os.path.join(DEFAULT_OUT_DIR,f"{exp_tag}.csv")

    config.set(conf.SEC_SIM, conf.EDGE_EXPOSED_FRACTION, "1.0")
    config.set(conf.SEC_SIM, conf.EDGE_NEIGHBORS, "100")
    config.set(conf.SEC_POLICY, conf.CLOUD_COLD_START_EST_STRATEGY, "pacs")
    config.set(conf.SEC_POLICY, conf.EDGE_COLD_START_EST_STRATEGY, "pacs")
    config.set(conf.SEC_POLICY, conf.POLICY_UPDATE_INTERVAL, "120")
    config.set(conf.SEC_POLICY, conf.POLICY_ARRIVAL_RATE_ALPHA, "0.3")
    config.set(conf.SEC_POLICY, conf.LOCAL_COLD_START_EST_STRATEGY, "naive-per-function")


    POLICIES = ["basic", "basic-edge", "probabilistic-legacy", "probabilistic", "probabilisticAlt"]

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
        for latency in [0.100, 0.200]:
            for n_edges in [1, 2, 5, 10, 20]:
                for pol in POLICIES:
                    config.set(conf.SEC_POLICY, conf.POLICY_NAME, pol)

                    keys = {}
                    keys["Policy"] = pol
                    keys["Seed"] = seed
                    keys["Latency"] = latency
                    keys["EdgeNodes"] = n_edges

                    run_string = "_".join([f"{k}{v}" for k,v in keys.items()])

                    # Check if we can skip this run
                    if old_results is not None and not\
                            old_results[(old_results.Seed == seed) &\
                                (old_results.Latency == latency) &\
                                (old_results.EdgeNodes == n_edges) &\
                                (old_results.Policy == pol)].empty:
                        print("Skipping conf")
                        continue

                    temp_spec_file = generate_temp_spec (n_edges=n_edges)
                    infra = default_infra(edge_cloud_latency=latency)
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

def experiment_simple (args, config):
    results = []
    exp_tag = "simple"
    outfile=os.path.join(DEFAULT_OUT_DIR,f"{exp_tag}.csv")

    config.set(conf.SEC_POLICY, conf.CLOUD_COLD_START_EST_STRATEGY, "pacs")
    config.set(conf.SEC_POLICY, conf.EDGE_COLD_START_EST_STRATEGY, "pacs")
    config.set(conf.SEC_POLICY, conf.POLICY_UPDATE_INTERVAL, "120")
    config.set(conf.SEC_POLICY, conf.POLICY_ARRIVAL_RATE_ALPHA, "0.3")


    POLICIES = ["basic", "basic-edge", "basic-budget", "probabilistic", "greedy-budget",  "probabilistic-strict",
                "probabilisticAlt", "probabilistic-strictAlt"]

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

def experiment_scalability (args, config):
    results = []
    exp_tag = "scalability"
    outfile=os.path.join(DEFAULT_OUT_DIR,f"{exp_tag}.csv")

    config.set(conf.SEC_SIM, conf.CLOSE_DOOR_TIME, str("400"))
    config.set(conf.SEC_POLICY, conf.LOCAL_COLD_START_EST_STRATEGY, "naive-per-function")
    config.set(conf.SEC_POLICY, conf.CLOUD_COLD_START_EST_STRATEGY, "naive-per-function")
    config.set(conf.SEC_POLICY, conf.EDGE_COLD_START_EST_STRATEGY, "no")
    config.set(conf.SEC_POLICY, conf.POLICY_UPDATE_INTERVAL, "120")
    config.set(conf.SEC_POLICY, conf.POLICY_ARRIVAL_RATE_ALPHA, "0.3")
    config.set(conf.SEC_POLICY, conf.HOURLY_BUDGET, "10")


    POLICIES = ["probabilistic-legacy", "probabilistic"]

    # Check existing results
    old_results = None
    if not args.force:
        try:
            old_results = pd.read_csv(outfile)
        except:
            pass

    for seed in SEEDS:
        config.set(conf.SEC_SIM, conf.SEED, str(seed))
        for n_classes in [1, 2, 4, 6, 8]:
            for functions in range(2,161,4):
                for pol in POLICIES:
                    config.set(conf.SEC_POLICY, conf.POLICY_NAME, pol)

                    keys = {}
                    keys["Policy"] = pol
                    keys["Seed"] = seed
                    keys["Functions"] = functions
                    keys["Classes"] = n_classes

                    run_string = "_".join([f"{k}{v}" for k,v in keys.items()])

                    # Check if we can skip this run
                    if old_results is not None and not\
                            old_results[(old_results.Seed == seed) &\
                                (old_results.Classes == n_classes) &\
                                (old_results.Functions == functions) &\
                                (old_results.Policy == pol)].empty:
                        print("Skipping conf")
                        continue

                    temp_spec_file = generate_temp_spec (n_functions=functions, n_classes=n_classes)
                    infra = default_infra()
                    stats = _experiment(config, infra, temp_spec_file.name)
                    temp_spec_file.close()
                    with open(os.path.join(DEFAULT_OUT_DIR, f"{exp_tag}_{run_string}.json"), "w") as of:
                        stats.print(of)
                    
                    
                    
                    result=dict(list(keys.items()) + list(relevant_stats_dict(stats).items()))

                    update_time = max([
                        stats._policy_update_time_sum[n]/stats._policy_updates[n] for n in infra.get_nodes()
                        ])
                    result["updateTime"] = update_time

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
    # if you 'force' old results will be deleted
    parser.add_argument('--force', action='store_true', required=False, default=False)
    # additional option to force results and seeds
    parser.add_argument('--debug', action='store_true', required=False, default=False)

    parser.add_argument('--seed', action='store', required=False, default=None, type=int)

    args = parser.parse_args()

    config = conf.parse_config_file("config.ini")
    config.set(conf.SEC_SIM, conf.STAT_PRINT_INTERVAL, "-1")
    config.set(conf.SEC_SIM, conf.CLOSE_DOOR_TIME, str(DEFAULT_DURATION))

    if args.debug:
        args.force = True
        SEEDS=SEEDS[:1]

    if args.seed is not None:
        SEEDS = [int(args.seed)]
    
    if args.experiment.lower() == "a":
        print("experiment to compare different policies and function configurations results")
        experiment_main_comparison(args, config)
    elif args.experiment.lower() == "b":
        print("experiment b")
        experiment_arrivals_to_all(args, config)
    elif args.experiment.lower() == "v":
        print("experiment v")
        experiment_varying_arrivals(args, config)
    elif args.experiment.lower() == "s":
        print("experiment s")
        experiment_scalability(args, config)
    elif args.experiment.lower() == "x":
        print("experiment x")
        experiment_simple(args, config)
    else:
        print("Unknown experiment!")
        exit(1)
