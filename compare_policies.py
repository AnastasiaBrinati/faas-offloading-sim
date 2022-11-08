import pandas as pd
import main
import conf

config = main.parse_config_file()
config.set(conf.SEC_SIM, conf.STAT_PRINT_INTERVAL, "-1")
config.set(conf.SEC_SIM, conf.CLOSE_DOOR_TIME, "120")

POLICIES = ["probabilistic", "probabilistic-legacy", "basic", "random"]
SEEDS = [(1,56), (2,23), (53, 98), (12,90), (567, 4)]

results = []
COL_NAMES = ["Policy", "Seeds", "Utility", "Cost"]


for policy in POLICIES:
    for s1,s2 in SEEDS:
        config.set(conf.SEC_SEED, conf.SEED_ARRIVAL, str(s1))
        config.set(conf.SEC_SEED, conf.SEED_SERVICE, str(s2))
        config.set(conf.SEC_POLICY, conf.POLICY_NAME, policy)
        simulation = main.init_simulation(config)
        stats = simulation.run()

        results.append((policy,(s1,s2),stats.utility, stats.cost))

df = pd.DataFrame(results, columns=COL_NAMES)
mean_df = df.groupby("Policy").mean()
std_df = df.groupby("Policy").std()
merged = pd.merge(mean_df, std_df, on="Policy", suffixes=["Mean","Std"])

print(merged)
with open("results_compare_policies.txt", "w") as of:
    print(merged, file=of)

