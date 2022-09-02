from pathlib import Path

STEPS_IN = 1
STEPS_OUT = 1
TRAIN_TEST_SPLIT = 0.8
EPOCHS = 100
BATCH_SIZE = 16
FIT_SHUFFLE = True
PREPROCESSING = "filtered-0.9_clr-1-True"

# Each dataset is preprocessed (filtered and clr-transformed) independently
DATA_PATH = Path("/storage/zkarwowska/microbiome-interactions/"
                 "datasets/processed/ready_datasets_preprocessed")
# MAIN_PATH = Path(__file__).parent / "results" / "baseline_diff"
MAIN_PATH = Path(__file__).parent / "results" / "baseline"

DATASETS = ['donorA', 'donorB', 'male', 'female']
DATASETS = [f"{d}_{PREPROCESSING}" for d in DATASETS]
SCALERS = ['id']
TRAIN_VAL_PARAMS = [
    {'overlap': False, 'shuffle': False},
    {'overlap': True, 'shuffle': False},
    {'overlap': False, 'shuffle': True},
    {'overlap': True, 'shuffle': True},
]

# Additional model's named arguments
KWARGS_SUP = [
    # BASIC
    {"use_bias": True, "L1": 0, "L2": 0},
    # # DEFAULT
    {"use_bias": True, "L1": 0.0001, "L2": 0.0001},
    # OTHER COMBINATIONS
    {"use_bias": False, "L1": 0.0001, "L2": 0.0001},
]
KWARGS_SEQ = [
    # BASIC
    {"use_input_bias": True, "use_pred_bias": True, "input_L1": 0, 
     "input_L2": 0, "pred_L1": 0, "pred_L2": 0},
    # DEFAULT
    {"use_input_bias": True, "use_pred_bias": True, "input_L1": 0.001, 
     "input_L2": 0.001, "pred_L1": 0.0001, "pred_L2": 0.0001},
    # OTHER COMBINATIONS
    {"use_input_bias": True, "use_pred_bias": True, "input_L1": 0.001,
     "input_L2": 0.001, "pred_L1": 0, "pred_L2": 0},
    {"use_input_bias": True, "use_pred_bias": True, "input_L1": 0, 
     "input_L2": 0, "pred_L1": 0.0001, "pred_L2": 0.0001},
    {"use_input_bias": False, "use_pred_bias": False, "input_L1": 0.001, 
     "input_L2": 0.001, "pred_L1": 0.0001, "pred_L2": 0.0001},
    {"use_input_bias": True, "use_pred_bias": False, "input_L1": 0, 
     "input_L2": 0, "pred_L1": 0.0001, "pred_L2": 0.0001},
    {"use_input_bias": False, "use_pred_bias": True, "input_L1": 0.001, 
     "input_L2": 0.001, "pred_L1": 0, "pred_L2": 0},
]


def _dict_to_str(dict_):
    if dict_:
         return '_' + '_'.join(['='.join(map(str, list(i))) for
                                i in dict_.items()])
    else:
        return ""
