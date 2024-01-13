import argparse
import gc
import os
import time
from sklearn.datasets import make_regression

import numpy as np
import pandas as pd

import ple_transformer 

NUM_FEATURES = 10  # Define NUM_FEATURES as a constant

# helper function to test the different implementations of matrix multiplication
def test_transformer(TransformerToTest, df, trial, num_bins=45):
    # create the transformer
    transformer_to_test = TransformerToTest(num_bins=num_bins)

    # disable garbage collection to get more accurate timing results
    gc.disable()

    # fit and transform the data
    start_time = time.time()
    transformer_to_test.fit(df)
    end_time = time.time()
    fit_duration = end_time - start_time

    start_time = time.time()
    transformer_to_test.transform(df)
    end_time = time.time()
    transform_duration = end_time - start_time

    # enable garbage collection again
    gc.enable()

    transformer_name = TransformerToTest.__name__
    short_name = transformer_name.replace("PiecewiseLinearEncoder", "")

    return {"transformer": short_name, "trial": trial, "num_samples": len(df), "fit_duration": fit_duration, "transform_duration": transform_duration}


if __name__ == "__main__":

    print(f"Running test scenarios... in {os.getcwd()}")
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the positional argument
    parser.add_argument("num_samples", type=int, nargs='?', default=1000, help="Number of samples to generate with a default value of 1000")
    parser.add_argument("num_features", type=int, nargs='?', default=10, help="Number of features to generate with default value of 10")
    parser.add_argument("transformer_class", type=str, nargs='?', default="MyClass", help="Name of the transformer class to test with default value of 'MyClass'")
    

    # Add the keyword argument
    parser.add_argument("--test_results_fp", type=str, default="test_results_multiple.csv", help="Specifies file path to save test reults with default value of 'test_results_multiple.csv'")

    # Parse the arguments
    args = parser.parse_args()

    # Now you can access the arguments as args.num and args.test_results_file
    print(args.num_samples)
    print(args.num_features)
    print(args.test_results_fp)
    print(args.transformer_class)
    test_results_fp = args.test_results_fp

    # Generate synthetic regression data
    X, y = make_regression(n_samples=args.num_samples, n_features=args.num_features, noise=0.1, random_state=1)

    # Convert to pandas DataFrame
    df = pd.DataFrame(data=X, columns=[f'Feature_{i}' for i in range(1, args.num_features+1)])
    df['Target'] = y

    df_data = df.drop('Target', axis=1).astype('float32')

    print(df_data.head())


    # run the test scenarios
    test_results = []
    print(f"Running test scenario for {args.transformer_class}")

    for trial in range(5):
        test_results.append(
            test_transformer(getattr(ple_transformer,args.transformer_class), df, trial)
        )

    # save the test results
    test_results_df = pd.DataFrame(test_results)
    if os.path.isfile(test_results_fp):
        # If the file exists, append without writing the header
        test_results_df.to_csv(test_results_fp, mode='a', header=False, index=False)
    else:
        # If the file does not exist, write the DataFrame to a new file with a header
        test_results_df.to_csv(test_results_fp, mode='w', header=True, index=False)

