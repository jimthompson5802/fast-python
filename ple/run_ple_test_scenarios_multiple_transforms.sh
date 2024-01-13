#!/bin/bash

rm test_results_multiple.csv

# array for values 1000, 10000
# declare -a num_samples=("1000")
declare -a num_samples=("400000" )

# array of transformer names
declare -a transformers=("PiecewiseLinearEncoderNumpy" "PiecewiseLinearEncoderNumbaV0" "PiecewiseLinearEncoderCython")

# loop trials
for i in {1..1}
do
    echo running trial $i
    # loop through the array of values
    for j in "${num_samples[@]}"
    do
        # loop through the array of transformer names
        for k in "${transformers[@]}"
        do
            # run the test scenarios
            python ple_test_scenarios_multiple_transforms.py $j 20 $k
        done
    done
done