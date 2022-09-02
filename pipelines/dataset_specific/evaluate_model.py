import sys
import pickle
import json
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow import keras

# TODO: solve later using setup.py
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.train_test import (series_to_supervised, split_reframed,
                              prepare_sequential_data, 
                              prepare_supervised_data)
from utils.evaluate import calculate_measures
from utils.transformers import inverse_through_timesteps_wrapper
from pipelines.dataset_specific.baseline_config import (DATA_PATH, MAIN_PATH,
                                                        DATASETS, _dict_to_str)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model.")

    parser.add_argument("-m", "--model_name", required=True, 
                        help="Name of the model e.g. 'mlp'")
    parser.add_argument("-i", "--model_input", required=False, 
                        default='supervised', help="Model input type "
                        "('supervised' or 'sequential')")
    parser.add_argument("-s", "--scaler_name", required=True, 
                        help="Scaler name e.g. 'minmax', 'clr_0_False'")
    parser.add_argument("-d", "--dataset_name", required=True, 
                        help="Dataset name e.g. 'donorA', 'male'")
    parser.add_argument("-t", "--train_val_params", required=True, 
                        type=json.loads,
                        help="Dictionary of train/val split parameters.'")
    parser.add_argument("-k", "--kwargs", required=True, type=json.loads,
                        help="Dictionary of additional named arguments.'")
    return parser.parse_args()


def main(): 
    
    # Inputs
    args = parse_args()
    mname = args.model_name
    itype = args.model_input
    sname = args.scaler_name  # NOT USED YET !!!
    dname = args.dataset_name
    train_val_params = args.train_val_params
    kwargs = args.kwargs

    print(f"\nRunning evaluation for: {mname}, {itype}, {sname}, {dname},"\
          f"{train_val_params}, {kwargs}")
    
    INPUT_PATH = MAIN_PATH /\
    f"{mname}_{itype}_{sname}_{dname}{_dict_to_str(train_val_params)}"\
    f"{_dict_to_str(kwargs)}"
    OUT_PATH = INPUT_PATH / "scores"
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    # Additional parameters for `scaler.inverse_transorm()`
    # For `clr` transformations we don't want to remove pseudocounts 
    # since it requires knowledge where initial 0's are located 
    # (there are not always known)

    sparams = {'remove_pseudocounts': False} if 'clr' in dname else {}
        
    # Load data
    
    model = keras.models.load_model(INPUT_PATH / 'model')
    # datasets are needed for inverse transform and naive prediction
    dataset = pd.read_csv(DATA_PATH / "filtered_transformed" /\
                          f"{dname}.csv", index_col=0)
    dname_original = '_'.join(dname.split('_')[:2])
    dataset_original = pd.read_csv(DATA_PATH / "filtered" /\
                                   f"{dname_original}.csv", index_col=0)  
    # transformer used for preprocessing
    transformer = joblib.load(DATA_PATH / f'scaler_{dname}.obj')
    data = np.load(INPUT_PATH / f'train_val_data.npz')
    
    model_config = json.load(open(INPUT_PATH / 'model_config.json', 'r'))
    STEPS_IN = model_config['input timesteps']
    STEPS_OUT = model_config['output timesteps']
    in_features = model_config['input features']
    
    # Make prediction
    if itype == 'sequential':
        # np.savez internally transforms lists into np.array
        # So, we have to transform it back into lists
        train_yhat = model.predict(list(data['train_X']))
        val_yhat = model.predict(list(data['val_X']))
    else:
        train_yhat = model.predict(data['train_X'])
        val_yhat = model.predict(data['val_X'])
    
    # Make naive prediction
    train_indices_y = data['train_indices_y']
    val_indices_y = data['val_indices_y']
    # - after preprocessing
    columns_dict = dict(zip(dataset, range(0, dataset.shape[1])))
    train_ynaive = dataset.loc[train_indices_y - 1].\
    rename(columns=columns_dict).set_index(train_indices_y)
    val_ynaive = dataset.loc[val_indices_y - 1].\
    rename(columns=columns_dict).set_index(val_indices_y)
    # - counts
    columns_dict = dict(zip(dataset_original, 
                            range(0, dataset_original.shape[1])))
    train_inv_ynaive = dataset_original.loc[train_indices_y - 1].\
    rename(columns=columns_dict).set_index(train_indices_y)
    val_inv_ynaive = dataset_original.loc[val_indices_y - 1].\
    rename(columns=columns_dict).set_index(val_indices_y)

    # Inverse data (preprocessing)    
    train_inv_y, val_inv_y = inverse_through_timesteps_wrapper(transformer, 
               dataset, [{'data': data['train_y'], 'index': train_indices_y}, 
               {'data': data['val_y'], 'index': val_indices_y}], sparams)
    train_inv_yhat, val_inv_yhat = inverse_through_timesteps_wrapper(transformer, 
               dataset, [{'data': train_yhat, 'index': train_indices_y}, 
               {'data': val_yhat, 'index': val_indices_y}], sparams)
       
    # Save raw predictions
    preds = {'trans_train_yhat': train_yhat, 'trans_val_yhat': val_yhat,
         'trans_train_ynaive': train_ynaive, 'trans_val_ynaive': val_ynaive,
         'trans_train_y': data['train_y'], 'trans_val_y': data['val_y'],
         'counts_train_yhat': train_inv_yhat, 'counts_val_yhat': 
         val_inv_yhat, 'counts_train_ynaive': train_inv_ynaive, 
         'counts_val_ynaive': val_inv_ynaive, 'counts_train_y': train_inv_y,
         'counts_val_y': val_inv_y}
    np.savez(INPUT_PATH / f'train_val_predictions', **preds)
    
    # Compute and save scores:
    scores_t = [{'return_tuple': True}, {'return_tuple': False}]
    models_t = ['yhat', 'ynaive']
    preds_t = ['trans', 'counts']
    data_t = ['train', 'val']
    for ind, score_t in enumerate(scores_t):
        for model_t in models_t:
            for pred_t in preds_t:
                for dat_t in data_t:
                    name = f"{pred_t}_{dat_t}_{model_t}"
                    scores = calculate_measures(preds[f"{pred_t}_{dat_t}_y"], 
                                        preds[name], **score_t)
                    if ind == 0:
                        json.dump(scores, open(OUT_PATH / f'{name}.json', 'w'))
                        print(f"{name}", scores, '\n')
                    else:
                        for k, v in scores.items():
                            v.to_csv(open(OUT_PATH / f'{name}_{k}.csv', 'w'))

                
if __name__ == '__main__':
    main()