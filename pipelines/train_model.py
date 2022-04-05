import sys
import pickle
import json
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# TODO: solve later using setup.py
sys.path.append(str(Path(__file__).parent.parent))

from utils.transformers import (CLRTransformer, Log1pMinMaxScaler, 
                                IdentityScaler)
from utils.train_test import (series_to_supervised, split_reframed,
                              prepare_sequential_data, 
                              prepare_supervised_data)
from models.baseline import (naive_predictor, sequential_mlp, 
                             supervised_mlp)
from pipelines.baseline_config import (STEPS_IN, STEPS_OUT, TRAIN_TEST_SPLIT,
                                       EPOCHS, BATCH_SIZE, TRAIN_SHUFFLE,
                                       FIT_SHUFFLE, DATA_PATH, MAIN_PATH,
                                       _dict_to_str)


def parse_args():
    parser = argparse.ArgumentParser(description="Train model.")

    parser.add_argument("-m", "--model_name", required=True, 
                        help="Name of the model e.g. 'mlp', 'naive'")
    parser.add_argument("-i", "--model_input", required=False, 
                        default='supervised', help="Model input type "
                        "('supervised' or 'sequential')")
    parser.add_argument("-s", "--scaler_name", required=True, 
                        help="Scaler name e.g. 'minmax', 'clr_0_False'")
    parser.add_argument("-d", "--dataset_name", required=True, 
                        help="Dataset name e.g. 'donorA', 'male'")
    parser.add_argument("-k", "--kwargs", required=True, type=json.loads,
                        help="Dictionary of additional named arguments.'")
    return parser.parse_args()


def main():
        
    # Inputs
    args = parse_args()
    mname = args.model_name
    itype = args.model_input
    sname = args.scaler_name
    dname = args.dataset_name
    kwargs = args.kwargs
       
    print(f"Training model for: {mname}, {itype}, {sname}, {dname}, {kwargs}")

    OUT_PATH = MAIN_PATH /\
    f"{mname}_{itype}_{sname}_{dname}_{_dict_to_str(kwargs)}"
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    # Load data

    dataset = pd.read_csv(DATA_PATH / f"{dname}_{sname}.csv", index_col=0)
    dataset_original = pd.read_csv(DATA_PATH / f"{dname}.csv", index_col=0)
    scaler = joblib.load(DATA_PATH / f'scaler_{dname}_{sname}.obj')
        
    assert dataset.shape == dataset_original.shape

    columns = [1, 5, 100, 150]
    plt.figure(figsize=(10, 7))
    plt.suptitle(f"{sname}, {dname}")
    for i, column in enumerate(columns):
        plt.subplot(len(columns), 1, i+1)
        plt.plot(dataset.iloc[:, column])
        # plt.plot(dataset_original.iloc[:, column])
        plt.title(column, y=0.5, loc='right')
    plt.tight_layout()
    plt.savefig(OUT_PATH / f"scaler_example.png")
    
    # Prepare training / validation data
    
    reframed = series_to_supervised(dataset.values, STEPS_IN, STEPS_OUT)
    train_X, train_y, test_X, test_y = split_reframed(reframed, 
                                                      len(dataset.columns), 
                                                      TRAIN_TEST_SPLIT, 
                                                      STEPS_IN,
                                                      shuffle=TRAIN_SHUFFLE)
    print(f"\nDataset shape: {dataset.shape}")
    print(f"Reframed shape: {reframed.shape}")
    print("\nInitial shapes:")
    print(f"Train_X shape: {train_X.shape}")
    print(f"Train_y shape: {train_y.shape}")
    print(f"Test_X shape: {test_X.shape}")
    print(f"Test_y shape: {test_y.shape}")
    
    in_steps = train_X.shape[1]
    in_features = train_X.shape[2]
    out_features = train_y.shape[1]
    
    assert in_steps == STEPS_IN
    assert in_features == dataset.shape[1]
    assert out_features == dataset.shape[1] * STEPS_OUT
    
    print("\nFinal shapes:")
    if itype == 'supervised':
        train_X, test_X = prepare_supervised_data(train_X, test_X)
        print(f"Train_X shape: {train_X.shape}")
        print(f"Test_X shape: {test_X.shape}")
    elif itype == 'sequential':
        train_X, test_X = prepare_sequential_data(train_X, test_X, 
                                                  in_features)
        print(f"Train_X shape: {len(train_X), train_X[0].shape}")
        print(f"Test_X shape: {len(test_X), test_X[0].shape}")
    else:
        raise NotImplementedError

    # Load model
    
    pred_activation = 'linear' if dataset.min().min() < 0 else 'relu'   
    
    if mname == 'mlp' and itype == 'supervised':
        model = supervised_mlp(in_steps, in_features, 
                               out_features, pred_activation=pred_activation,
                               **kwargs)    
    elif mname == 'mlp' and itype == 'sequential':
        model = sequential_mlp(in_steps, in_features, 
                               out_features, pred_activation=pred_activation,
                               **kwargs)
    elif mname == 'naive' and itype == 'supervised':
        model = naive_predictor('sup', in_features, STEPS_IN, STEPS_OUT) 
    elif mname == 'naive' and itype == 'sequential':
        model = naive_predictor('seq', in_features, STEPS_IN, STEPS_OUT) 
    else:
        raise NotImplementedError
    
    if mname != 'naive':

        # Fit model

        history = model.fit(train_X, train_y, epochs=EPOCHS, 
                            validation_data=(test_X, test_y), 
                            batch_size=BATCH_SIZE, verbose=0, 
                            shuffle=FIT_SHUFFLE)

        # Plot history

        plt.figure()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f"{mname}, {itype}, {sname}, {dname}")
        plt.savefig(OUT_PATH / f"fit.png")

        # Save model

        model.save(OUT_PATH / f'model.h5')
    else:
        with open(OUT_PATH / 'model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
    # Save traininig / validation data for later evaluation
    
    data = {'train_X': train_X, 'train_y': train_y,
            'val_X': test_X, 'val_y': test_y}
    np.savez(OUT_PATH / f'train_val_data', **data)
    
    # Create and save output config
    
    model_config = {
        'dataset name': dname,
        'scaler name': sname,
        'model name': mname,
        'input type': itype,
        'model params': model.count_params(),
        'prediction layer activation': pred_activation,
        'input timesteps': STEPS_IN,
        'output timesteps': STEPS_OUT,
        'input features': in_features,
        'train/validation split': TRAIN_TEST_SPLIT,
        'epochs': EPOCHS,
        'batch size': BATCH_SIZE,
        'train shuffle': TRAIN_SHUFFLE,
        'fit shuffle': FIT_SHUFFLE,
        **kwargs
    }
    
    json.dump(model_config, open(OUT_PATH / 'model_config.json', 'w'))

if __name__ == '__main__':
    main()