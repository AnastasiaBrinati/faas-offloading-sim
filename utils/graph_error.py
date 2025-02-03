import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
model = "model1.pkl_errors"
file_path = "results/errors/models/"+model+".csv"
df = pd.read_csv(file_path)

# Extract columns
errors = df["Error"]

df['media_cumulativa'] = errors.expanding().mean()

# Create index for x-axis
x_values = list(range(len(errors)))

# Plot
plt.figure(figsize=(21, 7))
plt.plot(x_values, df['media_cumulativa'], linestyle='-', color='blue', label='Actual')

# Labels and Title
plt.xlabel("Updates")
plt.ylabel("Error")
plt.title("Error: "+model)
plt.legend()
plt.grid(True)

# Show Plot
plt.savefig("results/errors/img/"+model+".png")
