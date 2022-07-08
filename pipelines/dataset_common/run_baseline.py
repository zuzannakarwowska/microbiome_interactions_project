"""
Run nested for loop over models, input types, scalers and datasets
for basic model configurations.

First, train models and save results.
Second, evaluate models and save results.

The output might be further processed with visualization tools.

Run the script (from the main repo directory) e.g. as:
    nohup python pipelines/dataset_common/run_baseline.py \
    > pipelines/dataset_common/results/baseline.log \
    2> pipelines/dataset_common/results/baseline.err
"""


import sys
import subprocess
import json

from pathlib import Path

# TODO: solve later using setup.py
sys.path.append(str(Path(__file__).parent.parent.parent))

from pipelines.dataset_common.baseline_config import (MAIN_PATH, DATASETS, 
                                                      SCALERS, KWARGS_SUP, 
                                                      KWARGS_SEQ, _dict_to_str)


MODELS = ['mlp', 'mlp_diff', 'naive']
INPUT_TYPES = ['supervised', 'sequential']
    
    
if __name__ == '__main__':

    LOG_PATH = MAIN_PATH / "log"
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    
    for model in MODELS:
        for input_type in INPUT_TYPES:
            
            # arguments list depends on input type
            if input_type == 'supervised':
                kwargs_list = KWARGS_SUP
            if input_type == 'sequential':
                kwargs_list = KWARGS_SEQ
                
            for scaler in SCALERS:
                for dataset in DATASETS:
                    for kwargs in kwargs_list:
                        name = f"{model}_{input_type}_{scaler}"\
                        f"_{dataset}{_dict_to_str(kwargs)}"
                        print(f"Processing: {name}")
        
                        # Training
                        f_out = open(LOG_PATH / f"{name}.log", "w")
                        f_err = open(LOG_PATH / f"{name}.err", "w")
                        subprocess.call(["python", 
                                             "pipelines/dataset_common/"\
                                             "train_model.py",
                                         "-m", model, "-i", input_type, 
                                         "-s", scaler, "-d", dataset,
                                         "-k", json.dumps(kwargs)], 
                                        stdout=f_out, stderr=f_err)

                        # Evaluation
                        f_out = open(LOG_PATH / f"{name}.log", "a")
                        f_err = open(LOG_PATH / f"{name}.err", "a")
                        subprocess.call(["python", 
                                             "pipelines/dataset_common/"\
                                             "evaluate_model.py",
                                         "-m", model, "-i", input_type, 
                                         "-s", scaler, "-d", dataset, 
                                         "-k", json.dumps(kwargs)], 
                                        stdout=f_out, stderr=f_err)
                   
