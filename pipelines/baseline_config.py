from pathlib import Path

STEPS_IN = 1
STEPS_OUT = 1
TRAIN_TEST_SPLIT = 0.8
EPOCHS = 100
BATCH_SIZE = 16
TRAIN_OVERLAP = True
TRAIN_SHUFFLE = True
FIT_SHUFFLE = True

DATA_PATH = Path("/storage/zkarwowska/microbiome-interactions/"
                 "datasets/processed/ready_datasets_transformed/common")
MAIN_PATH = Path(__file__).parent / "results" / "baseline_bias_reg"

DATASETS = ['donorA', 'donorB', 'male', 'female']
# SCALERS = ['id', 'std', 'minmax', 'quantile10', 'quantile50', 'quantile100', 
#            'quantile150', 'clr_0_False', 'clr_0_True', 'clr_None_False', 
#            'clr_None_True', 'log1pminmax']
SCALERS = ['id', 'std', 'clr_0_True']

# Additional model's named arguments
KWARGS_SUP = [
    {"use_bias": True, "L1": 0.0001, "L2": 0.0001},
    {"use_bias": False, "L1": 0.0001, "L2": 0.0001},
]
KWARGS_SEQ = [
    {"use_input_bias": True, "use_pred_bias": True, "input_L1": 0.001, 
     "input_L2": 0.001, "pred_L1": 0.0001, "pred_L2": 0.0001},
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
    return '_'.join(['='.join(map(str, list(i))) for i in dict_.items()])