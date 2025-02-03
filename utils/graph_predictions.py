import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
edge = "edge3_predictions"
file_path = "results/predictions/"+edge+".csv"
df = pd.read_csv(file_path)

# Extract columns
actual = df["Actual"]
predicted = df["Predicted"]

# Create index for x-axis
x_values = list(range(len(actual)))

# Plot
plt.figure(figsize=(21, 7))
plt.plot(x_values, actual, linestyle='-', color='blue', label='Actual')
plt.plot(x_values, predicted, linestyle='--', color='red', label='Predicted')

# Labels and Title
plt.xlabel("Time")
plt.ylabel("Arrival Rate")
plt.title("Actual vs. Predicted Arrival Rates")
plt.legend()
plt.grid(True)

# Show Plot
plt.savefig("results/predictions/img/"+edge+".png")
