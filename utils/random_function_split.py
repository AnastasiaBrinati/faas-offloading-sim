import pandas as pd
import numpy as np

np.random.seed(123)

def cluster(data,title, n_clusters=2):
    # Step 1: Load the data into a DataFrame
    df = pd.DataFrame(data)

    # Step 2: Randomly assign each row to a cluster
    df['cluster'] = np.random.randint(0, n_clusters, size=len(df))

    # Step 3: Save each cluster into a separate file
    for cluster_id in range(n_clusters):
        cluster_df = df[df['cluster'] == cluster_id].drop(columns=['cluster'])
        filename = f"traces/synthetic_{title}_f{cluster_id}.csv"
        cluster_df.to_csv(filename, index=False)
        print(f"Saved {filename} with {len(cluster_df)} records.")


if __name__ == "__main__":
    title = "logistic-map"
    data = pd.read_csv("traces/synthetic/synthetic_"+title+"_arrivals.csv")
    cluster(data, title, n_clusters=2)