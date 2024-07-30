#!/bin/bash

# Define the path to the CSV file
csv_file="list_nodes.csv"

# Check if the CSV file exists
if [ ! -f "$csv_file" ]; then
    echo "CSV file not found: $csv_file"
    exit 1
fi

# Flag to skip the first line
skip_first_line=true

# Loop through each line in the CSV file
while IFS=, read -r column1 column2 rest_of_columns; do
    # Skip the first line
    if $skip_first_line; then
        skip_first_line=false
        continue
    fi

    # Run the Python script with the value from the first column
    echo "Deleting node  $column2"
    yes y | verdi node delete "$column2"
done < "$csv_file"
