"""
Run nested for loop over models, input types, scalers and datasets
for basic model configurations.

First, train models and save results.
Second, evaluate models and save results.

The output might be further processed with visualization tools.

Run the script (from the main repo directory) e.g. as:
    nohup python pipelines/run_baseline.py \
    > pipelines/results/baseline.log \
    2> pipelines/results/baseline.err
"""


import sys
import subprocess

from pathlib import Path

# TODO: solve later using setup.py
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.baseline_config import MAIN_PATH, DATASETS, SCALERS


MODELS = ['naive', 'mlp']
INPUT_TYPES = ['supervised', 'sequential']


if __name__ == '__main__':

    LOG_PATH = MAIN_PATH / "log"
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    
    for model in MODELS:
        for input_type in INPUT_TYPES:
            for scaler in SCALERS:
                for dataset in DATASETS:
                    
                    name = f"{model}_{input_type}_{scaler}_{dataset}"
                    print(f"Processing: {name}")
    
                    # Training
                    # f_out = open(LOG_PATH / f"{name}.log", "w")
                    # f_err = open(LOG_PATH / f"{name}.err", "w")
                    # subprocess.call(["python", "pipelines/train_model.py",
                    #                  "-m", model, "-i", input_type, 
                    #                  "-s", scaler, "-d", dataset], 
                    #                 stdout=f_out, stderr=f_err)

                    # Evaluation
                    f_out = open(LOG_PATH / f"{name}.log", "a")
                    f_err = open(LOG_PATH / f"{name}.err", "a")
                    subprocess.call(["python", "pipelines/evaluate_model.py",
                                     "-m", model, "-i", input_type, 
                                     "-s", scaler, "-d", dataset], 
                                    stdout=f_out, stderr=f_err)