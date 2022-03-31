from pathlib import Path

STEPS_IN = 1
STEPS_OUT = 1
TRAIN_TEST_SPLIT = 0.8
EPOCHS = 100
BATCH_SIZE = 16
TRAIN_SHUFFLE = True
FIT_SHUFFLE = True

DATA_PATH = Path("/storage/zkarwowska/microbiome-interactions/"
                 "datasets/processed/ready_datasets_transformed/common")
MAIN_PATH = Path(__file__).parent / "results" / "baseline_scalers"

DATASETS = ['donorA', 'donorB', 'male', 'female']
SCALERS = ['id', 'std', 'minmax', 'quantile10', 'quantile50', 'quantile100', 
           'quantile150', 'clr_0_False', 'clr_0_True', 'clr_None_False', 
           'clr_None_True', 'log1pminmax']