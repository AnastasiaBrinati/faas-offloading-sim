#!/bin/bash

# File paths
SPEC_FILE="spec.yml"
CONFIG_FILE="config.ini"
OUTPUT_FILE="results_finals.csv"
# Backup file
BACKUP_FILE="${SPEC_FILE}.bak"

# Policy names to iterate over
POLICIES=("probabilistic-function" "predictive-function")
# Definizione delle classi disponibili
CLASSES=("all-classes" "best-effort" "critical" "deferrable")

# Scrive l'intestazione del file di output (se non esiste)
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "policy,f1_class,f2_class,f3_class,cost,utility" > "$OUTPUT_FILE"
fi

# Iterate over all combinations of class1, class2, class3, and the "all-classes" option
for class1 in "${CLASSES[@]}"; do
  for class2 in "${CLASSES[@]}"; do
    for class3 in "${CLASSES[@]}"; do
        # Restore the original spec.yml from the backup
        cp "$BACKUP_FILE" "$SPEC_FILE"

        # Modify the classes in the arrivals section for each function (f1, f2, f3) using `sed`
        if [[ "$class1" == "all-classes" ]]; then
            sed -i '/function: f1/ {n; d; n; d; }' "$SPEC_FILE"
        else
            sed -i "/function: f1/{n;s|classes:.*|classes:|; n; s|$|    - $class1|; }" "$SPEC_FILE"
        fi

        if [[ "$class2" == "all-classes" ]]; then
            sed -i '/function: f2/ {n; d; n; d; }' "$SPEC_FILE"
        else
            sed -i "/function: f2/{n;s|classes:.*|classes:|; n; s|$|    - $class2|; }" "$SPEC_FILE"
        fi

        if [[ "$class3" == "all-classes" ]]; then
            sed -i '/function: f3/ {n; d; n; d; }' "$SPEC_FILE"
        else
            echo "setting class3 to: $class3"
            sed -i "/function: f3/{n;s|classes:.*|classes:|; n; s|$|    - $class3|; }" "$SPEC_FILE"
        fi

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
            echo "$policy,$class1,$class2,$class3,$cost,$utility" >> "$OUTPUT_FILE"
        done

    done
  done
done
