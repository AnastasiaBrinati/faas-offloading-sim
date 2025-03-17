#!/bin/bash

# Policy names to iterate over
POLICIES=("probabilistic-function" "predictive-function" "online-predictive-function" "online-adaptive-function")
DURATIONS=("0.400")
#MEMORIES=("128")
CONFIG_FILE="config.ini"
SPEC_FILE="spec.yml"

OUTPUT_FILE="results2.csv"
# Scrive l'intestazione del file di output (se non esiste)
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "function_mean_duration,policy,cost,utility" > "$OUTPUT_FILE"
fi

#for memory in "${MEMORIES[@]}"; do
  #sed -i "s/memory: [0-9]*\.[0-9]*/memory: $memory/" "$SPEC_FILE"
for duration in "${DURATIONS[@]}"; do
  sed -i "s/duration_mean: [0-9]*\.[0-9]*/duration_mean: $duration/" "$SPEC_FILE"

  # Iterate over both policy names
  for policy in "${POLICIES[@]}"; do

      # Modify config.ini to set the current policy
      sed -i "/\[policy\]/,/^\[/ s/^name = .*/name = $policy/" "$CONFIG_FILE"
      echo "Set policy to: $policy"

      # Run the Python script and capture the output
      output=$(python3 main.py)

      # Extract cost and utility from the output
      cost=$(echo "$output" | grep "cost:" | awk '{print $2}')
      utility=$(echo "$output" | grep "utility:" | awk '{print $2}')

      # Save results in CSV
      echo "$duration,$policy,$cost,$utility" >> "$OUTPUT_FILE"
  done
done

python3 results/predictions/comparison_graph.py