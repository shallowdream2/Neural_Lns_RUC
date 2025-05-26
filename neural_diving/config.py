# Config for Neural_Lns_RUC project

# Paths
DATA_DIR = "mediate_data"
MODEL_DIR = "models"

# train_diving_gcn.py hyperparameters
TRAIN_INPUT_DIM = 6
TRAIN_HIDDEN_DIM = 128
TRAIN_OUTPUT_DIM = 1
TRAIN_N_EPOCHS = 100
TRAIN_LR = 1e-2
TRAIN_N_BITS = 8 # Assuming n_bits from DivingGCN init

# preprocessing.py parameters
PREPROC_MAX_SOLUTIONS = 100
PREPROC_TIME_LIMIT = 120
PREPROC_NUM_WORKERS = 1
PREPROC_SUMMARY_FILE = "preprocess_summary.json"
PREPROC_OUTPUT_FILE = None # Set to None to use dynamic naming

# build_graph_dataset.py parameters
BUILD_GRAPH_INPUT_FILE = None # Set to None to use default "training_data.pkl"
GRAPH_DATASET_OUTPUT_FILE = None # Set to None to use dynamic naming

# train_diving_gcn.py parameters
TRAIN_INPUT_FILE = None # Set to None to use the latest *_graph.pkl file
TRAIN_MODEL_SAVE_NAME = "diving_gcn_mediate.pt" # Set to None to disable saving or use default

# evaluate_model.py parameters
EVAL_DATA_INPUT_FILE = "20250526_164612_graph.pkl" # Set to None to use the latest *_graph.pkl file
EVAL_MODEL_INPUT_FILE = "models/diving_gcn_test.pt" # Set to None to use default "models/diving_gcn.pt" 