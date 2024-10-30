# ####### FILE PATHS #######

# The root folder where all subsequent relative paths will start from.
root_folder = "F:/Paolo/data_folder/"

# The folder where the results of the experiments will be saved.
result_folder = "F:/Paolo/results/"

# ####### DATA PARAMETERS #######

# List of feature column names to be used as input features for the model.
input_features = [
    'energy_1', 'energy_2', 'energy_3', 'energy_4', 'energy_5',
    'energy_6', 'energy_7', 'energy_8', 'energy_9', 'energy_10',
    'energy_11', 'energy_12'
]

# The column name of the target variable to be predicted by the model.
target_variable = 'energy_target'

# List of columns to be dropped from the dataset, if any.
drop_columns = ['cliente']

# File paths for the source domain datasets used in the experiments.
source_domain1 = root_folder + "source_domain_1/train_2012.csv"
source_domain2 = root_folder + "source_domain_2/train_2012.csv"

# Optional: Additional source domains can be added by defining their file paths.
# source_domain3 = ...
# source_domain4 = ...

# List of all source domain file paths.
source_domain_file_paths = [source_domain1, source_domain2]  # Additional domains can be added here.

# File paths for the target domain's training and testing datasets.
target_domain_train = root_folder + "target_domain/train_2012.csv"
target_domain_test = root_folder + "target_domain/test_2012.csv"

# ####### MODEL PARAMETERS #######

# Maximum number of iterations for the MLP model (note: early stopping is enabled,
# so the model may stop before reaching this number to avoid overfitting).
max_iter = 500

# Whether to reuse the solution of the previous call to fit the model and add more estimators to the ensemble (fine-tuning).
warm_start = False

# The seed for deterministic random processes, ensuring reproducibility.
seed = 123

# ####### EXPERIMENT PARAMETERS #######

# The name of the method or algorithm to be used in the experiments.
method_name = 'stacking'
