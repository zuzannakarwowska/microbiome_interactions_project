import sys
import pickle
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow import keras

# TODO: solve later using setup.py
sys.path.append(str(Path(__file__).parent.parent))

from utils.train_test import (series_to_supervised, split_reframed,
                              prepare_sequential_data, 
                              prepare_supervised_data)
from utils.evaluate import calculate_measures
from pipelines.baseline_config import DATA_PATH, MAIN_PATH, DATASETS


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model.")

    parser.add_argument("-m", "--model_name", required=True, 
                        help="Name of the model e.g. 'mlp', 'naive'")
    parser.add_argument("-i", "--model_input", required=False, 
                        default='supervised', help="Model input type "
                        "('supervised' or 'sequential')")
    parser.add_argument("-s", "--scaler_name", required=True, 
                        help="Scaler name e.g. 'minmax', 'clr_0_False'")
    parser.add_argument("-d", "--dataset_name", required=True, 
                        help="Dataset name e.g. 'donorA', 'male'")
    return parser.parse_args()


def main():
    
    # Inputs
    args = parse_args()
    mname = args.model_name
    itype = args.model_input
    sname = args.scaler_name
    dname = args.dataset_name

    print(f"Running evaluation for: {mname}, {itype}, {sname}, {dname}")
    
    INPUT_PATH = MAIN_PATH / f"{mname}_{itype}_{sname}_{dname}"
    OUT_PATH = INPUT_PATH / "scores"
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    # Additional parameters for `scaler.inverse_transorm()`
    # For `clr` transformations we don't want to remove pseudocounts 
    # since it requires knowledge where initial 0's are located 
    # (there are not always known)

    if sname.startswith('clr'):
        sparams = {'remove_pseudocounts': False} 
    else:
        sparams = {}
        
    # Load data
    
    if mname != 'naive':
        model = keras.models.load_model(INPUT_PATH / f'model.h5')
    else:
        model = pickle.load(open(INPUT_PATH / 'model.pkl', 'rb'))
    
    scaler = joblib.load(DATA_PATH / f'scaler_{dname}_{sname}.obj')
    data = np.load(INPUT_PATH / f'train_val_data.npz')
    
    model_config = pickle.load(open(INPUT_PATH / 'model_config.pkl', 'rb'))
    STEPS_IN = model_config['input timesteps']
    STEPS_OUT = model_config['output timesteps']
    in_features = model_config['input features']
    
    # Train / validation datasets
    
    # Make prediction
    if itype == 'sequential':
        # np.savez internally transforms lists into np.array
        # So, we have to transform it back into lists
        train_yhat = model.predict(list(data['train_X']))
        val_yhat = model.predict(list(data['val_X']))
    else:
        train_yhat = model.predict(data['train_X'])
        val_yhat = model.predict(data['val_X'])
    # Inverse data
    train_inv_y = scaler.inverse_transform(data['train_y'], **sparams)
    val_inv_y = scaler.inverse_transform(data['val_y'], **sparams)
    train_inv_yhat = scaler.inverse_transform(train_yhat, **sparams)
    val_inv_yhat = scaler.inverse_transform(val_yhat, **sparams)
    # Compute and save scores
    scores_train = calculate_measures(train_inv_y, train_inv_yhat)
    scores_val = calculate_measures(val_inv_y, val_inv_yhat)
    pickle.dump(scores_train, open(OUT_PATH / f'{dname}_train.pkl', 'wb'))
    pickle.dump(scores_val, open(OUT_PATH / f'{dname}_val.pkl', 'wb'))
    print(f"{dname} (train)", scores_train, '\n')
    print(f"{dname} (val)", scores_val, '\n')
        
    # Test datasets

    for test_dname in DATASETS:
        test_dataset_original = pd.read_csv(DATA_PATH / f"{test_dname}.csv", 
                                           index_col=0)

        test_dataset =  pd.DataFrame(scaler.transform(test_dataset_original), 
                                columns=test_dataset_original.columns, 
                                index=test_dataset_original.index)
        test_reframed = series_to_supervised(test_dataset.values, 
                                            STEPS_IN, STEPS_OUT)

        test_X, test_y, _, _ = split_reframed(test_reframed, 
                                            len(test_dataset.columns), 
                                            1, STEPS_IN, shuffle=False)

        if itype == 'supervised':
            test_X, _ = prepare_supervised_data(test_X)
            print(f"Test shape ({test_dname}): {test_X.shape}")
        elif itype == 'sequential':
            test_X, _ = prepare_sequential_data(test_X, None, in_features)
            print(f"Test shape ({test_dname}): {len(test_X), test_X[0].shape}")
        else:
            raise NotImplementedError

        # Make prediction
        test_yhat = model.predict(test_X)
        # Inverse data
        test_inv_y = scaler.inverse_transform(test_y, **sparams)
        test_inv_yhat= scaler.inverse_transform(test_yhat, **sparams)
        # Compute and save scores
        scores_test = calculate_measures(test_inv_y, test_inv_yhat)
        pickle.dump(scores_test, open(OUT_PATH / f'{test_dname}.pkl', 'wb'))
        print(test_dname, scores_test, '\n')

        
if __name__ == '__main__':
    main()