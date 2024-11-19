import math

def warn(*args, **kwargs):
    # Custom warning function that does nothing (to suppress warnings).
    pass

import warnings

# Override the default warning function with the custom warning function.
warnings.warn = warn

import datetime
import os

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

import parameters

import numpy as np

# Determine the directory of the current script.
path = os.path.dirname(os.path.abspath(__file__))

def run(source_domains, target_domain_train, target_domain_test):
    # Initialize lists to store training data from source domains.
    X_train_sources = []
    y_train_sources = []

    # Process each source domain file.
    for source_file in source_domains:
        # Extract the domain name from the file name.
        source_domain = source_file.split('/')[-1].split('.')[0]  # Assumes file name is in the format 'domainname.csv'

        # Read the CSV file into a DataFrame, and drop columns specified in parameters.
        source_df = pd.read_csv(source_file).drop(columns=parameters.drop_columns)
        y_train = source_df[parameters.target_variable]
        X_train = source_df.drop(columns=[parameters.target_variable])

        # Append processed data to the lists.
        X_train_sources.append((source_domain, X_train))
        y_train_sources.append((source_domain, y_train))

    # Extract the target domain name from the target training file path.
    target_domain = target_domain_train.split('/')[-1].split('.')[0]  # Assumes file name is in the format 'domainname.csv'

    # Read the target domain training CSV file into a DataFrame.
    target_train_df = pd.read_csv(target_domain_train).drop(columns=parameters.drop_columns)
    y_train = target_train_df[parameters.target_variable]

    # Compute the mean of the target variable for adjusted R-squared calculation.
    mean_y_train = y_train.mean()

    # Extract the input features for the target domain training set.
    X_train = target_train_df.drop(columns=[parameters.target_variable])

    # Read the target domain testing CSV file into a DataFrame.
    target_test_df = pd.read_csv(target_domain_test).drop(columns=parameters.drop_columns)
    X_test = target_test_df.drop(columns=[parameters.target_variable])
    y_test = target_test_df[parameters.target_variable]

    # Calculate hidden layer size for the MLPRegressor.
    hidden_layer_size = math.ceil((X_train.shape[1] + 1) / 2)

    # Generate a configuration folder path based on domain names and parameters.
    source_domains = "_".join([domain_name for domain_name, _ in X_train_sources])

    # Initialize the MLPRegressor model with specified parameters.
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_size, activation="relu", solver='adam',
        alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
        power_t=0.5, shuffle=True, random_state=parameters.seed, tol=0.0001, verbose=False,
        warm_start=parameters.warm_start, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
        epsilon=1e-08, n_iter_no_change=10, max_fun=15000, max_iter=parameters.max_iter
    )

    # Lists to collect predictions for each source domain.
    y_predicted_train_sources = []
    y_predicted_test_sources = []

    for i, ((domain_name_X, X_train_source), (domain_name_y, y_train_source)) in enumerate(zip(X_train_sources, y_train_sources)):
        # Ensure domain names match between X and y sources.
        assert domain_name_X == domain_name_y, "Domain names must match between X and y sources"

        # Fit the model on source domain data and make predictions.
        model_SOURCE = model.fit(X_train_source, y_train_source)
        y_predicted_train_source = model_SOURCE.predict(X_train)
        y_predicted_test_source = model_SOURCE.predict(X_test)

        # Store predictions in DataFrames with column names indicating source domain.
        y_predicted_train_sources.append(pd.DataFrame(y_predicted_train_source, columns=[f'y_predicted_source{i + 1}']).reset_index(drop=True))
        y_predicted_test_sources.append(pd.DataFrame(y_predicted_test_source, columns=[f'y_predicted_source{i + 1}']).reset_index(drop=True))

    # Train the model on the target domain data and measure training time.
    start_time = datetime.datetime.now()
    model_TARGET = model.fit(X_train, y_train)
    num_effective_iterations = model.n_iter_
    end_time = datetime.datetime.now()
    target_time = str((end_time - start_time).total_seconds())
    print(target_domain + "\t" + target_time + "\t" + str(num_effective_iterations))

    # Make predictions on target domain training and testing sets.
    y_predicted_train_target = model_TARGET.predict(X_train)
    y_predicted_test_target = model_TARGET.predict(X_test)

    # Concatenate predictions from all source domains and the target domain for stacking.
    stacking_multi_train_input_space = pd.concat(
        [pd.DataFrame(X_train).reset_index(drop=True)] + y_predicted_train_sources + [pd.DataFrame(y_predicted_train_target, columns=['y_predicted_target']).reset_index(drop=True)],
        axis=1
    ).reset_index(drop=True)
    print(stacking_multi_train_input_space.head())

    # Calculate hidden layer size for the stacking model.
    hidden_layer_size = math.ceil((stacking_multi_train_input_space.shape[1] + 1) / 2)

    # Initialize and fit the stacking model.
    stacking_model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_size, activation="relu", solver='adam',
        alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
        power_t=0.5, shuffle=True, random_state=parameters.seed, tol=0.0001, verbose=False,
        warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10,
        max_fun=15000, max_iter=parameters.max_iter
    )

    start_time = datetime.datetime.now()
    model_STACKING = stacking_model.fit(stacking_multi_train_input_space, y_train)
    num_effective_iterations = stacking_model.n_iter_
    end_time = datetime.datetime.now()
    time = str((end_time - start_time).total_seconds())
    print(parameters.method_name +"\t" + time + "\t" + str(num_effective_iterations))

    # Concatenate predictions for the stacking model on the test set.
    stacking_multi_test_input_space = pd.concat(
        [pd.DataFrame(X_test).reset_index(drop=True)] + y_predicted_test_sources + [pd.DataFrame(y_predicted_test_target, columns=['y_predicted_target']).reset_index(drop=True)],
        axis=1
    ).reset_index(drop=True)
    print(stacking_multi_test_input_space.head())

    # Create directory for saving results if it does not exist.
    config_folder = (parameters.result_folder + source_domains + "_" + target_domain + "_" + str(hidden_layer_size) + "_" + str(parameters.max_iter))
    print(config_folder)
    if not os.path.exists(config_folder):
        os.makedirs(config_folder)

    # Make predictions using the stacking model.
    y_predicted_stacking = model_STACKING.predict(stacking_multi_test_input_space)


    # Print and save the results.
    print_results(y_test, y_predicted_stacking, mean_y_train, parameters.method_name, config_folder)

def print_results(actual, predicted, mean_y_train, method_name, config_folder):
    # Ensure the directory exists before saving results.
    os.makedirs(os.path.dirname(config_folder), exist_ok=True)
    filename = config_folder + "/" + method_name + "_actual_predicted.txt"

    print(method_name)

    # Save actual vs predicted values to a text file.
    with open(filename, "ab") as f:
        np.savetxt(f, np.c_[actual, predicted], fmt='%s', delimiter=',', comments='')

    # Calculate and print MAE, RMSE, MSE, and R-squared metrics.
    mae = mean_absolute_error(actual, predicted)
    mae_to_write = str(mae)
    print("mae:\t" + mae_to_write)
    mae_file = config_folder + "/" + method_name + "_MAE.txt"
    with open(mae_file, "a") as r:
        r.write(mae_to_write + "\n")

    rmse = root_mean_squared_error(actual, predicted)
    rmse_to_write = str(rmse)
    print("rmse:\t" + rmse_to_write)
    rmse_file = config_folder + "/" + method_name + "_RMSE.txt"
    with open(rmse_file, "a") as r:
        r.write(rmse_to_write + "\n")

    mse = mean_squared_error(actual, predicted)
    mse_to_write = str(mse)
    print("mse:\t" + mse_to_write)
    mse_file = config_folder + "/" + method_name + "_MSE.txt"
    with open(mse_file, "a") as r:
        r.write(mse_to_write + "\n")

    r_squared = r2_score(actual, predicted)
    r_squared_to_write = str(r_squared)
    print("r2:\t" + r_squared_to_write)
    r_squared_file = config_folder + "/" + method_name + "_R2.txt"
    with open(r_squared_file, "a") as r:
        r.write(r_squared_to_write + "\n")

    # Calculate adjusted R-squared.
    mean_y_train_vector = np.full_like(actual, mean_y_train)
    trick_mse = mean_squared_error(actual, mean_y_train_vector)
    adjusted_r_squared = 1 - (mse / trick_mse)
    adjusted_r_squared_to_write = str(adjusted_r_squared)
    print("adjusted r2:\t" + adjusted_r_squared_to_write)
    adjusted_r_squared_file = config_folder + "/" + method_name + "_adjusted_R2.txt"
    with open(adjusted_r_squared_file, "a") as r:
        r.write(adjusted_r_squared_to_write + "\n")


# Start the experiments by calling the run function with parameters.
run(parameters.source_domain_file_paths, parameters.target_domain_train, parameters.target_domain_test)
