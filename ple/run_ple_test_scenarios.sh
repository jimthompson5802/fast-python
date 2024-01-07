#!/bin/bash

rm test_results.csv

# array for values 1000, 10000
# declare -a num_samples=("1000")
declare -a num_samples=("100000" "200000" "400000" "800000" )

# array of transformer names
declare -a transformers=("PiecewiseLinearEncoderNumpy" "PiecewiseLinearEncoderNumbaV0" "PiecewiseLinearEncoderNumbaV1" "PiecewiseLinearEncoderCython")

# loop through the array of values
for i in "${num_samples[@]}"
do
    # loop through the array of transformer names
    for j in "${transformers[@]}"
    do
        # run the test scenarios
        python ple_test_scenarios.py $i 10 $j
    done
done

