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
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.train_test import (series_to_supervised, split_reframed,
                              prepare_sequential_data, 
                              prepare_supervised_data)
from models.baseline import SupervisedMLP, SequentialMLP
from models.baseline_with_diff import SupervisedDiffMLP, SequentialDiffMLP
from pipelines.dataset_specific.baseline_config import (STEPS_IN, STEPS_OUT, 
                                       TRAIN_TEST_SPLIT, EPOCHS, BATCH_SIZE, 
                                       FIT_SHUFFLE, DATA_PATH, MAIN_PATH, 
                                       _dict_to_str)


def parse_args():
    parser = argparse.ArgumentParser(description="Train model.")

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
       
    print(f"Training model for: {mname}, {itype}, {sname}, {dname},"\
          f"{train_val_params}, {kwargs}")

    OUT_PATH = MAIN_PATH /\
    f"{mname}_{itype}_{sname}_{dname}{_dict_to_str(train_val_params)}"\
    f"{_dict_to_str(kwargs)}"
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    # Load data
    
    dataset = pd.read_csv(DATA_PATH / "filtered_transformed" /\
                          f"{dname}.csv", index_col=0)
    dataset_original = pd.read_csv(DATA_PATH / "original" /\
                                   f"{dname.split('_')[0]}.csv", index_col=0)
    # transformer used for preprocessing
    transformer = joblib.load(DATA_PATH / f'scaler_{dname}.obj')
    
    columns = [1, 5, 100, 150]
    plt.figure(figsize=(10, 7))
    plt.suptitle(f"{sname}, {dname}")
    for i, column in enumerate(columns):
        plt.subplot(len(columns), 1, i+1)
        plt.plot(dataset.iloc[:, column])
        # plt.plot(dataset_original.iloc[:, column])
        plt.title(column, y=0.5, loc='right')
    plt.tight_layout()
    plt.savefig(OUT_PATH / f"transformer_example.png")
    
    # Prepare training / validation data
    
    reframed = series_to_supervised(dataset.values, STEPS_IN, STEPS_OUT)

    train_X, train_y, test_X, test_y, train_indices_y, test_indices_y =\
    split_reframed(reframed, len(dataset.columns), TRAIN_TEST_SPLIT, STEPS_IN,
    overlap=train_val_params['overlap'], shuffle=train_val_params['shuffle'], 
    return_indices=True) 
    
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
        model = SupervisedMLP(in_steps, in_features, 
                               out_features, pred_activation=pred_activation,
                               **kwargs)    
        model.compile(optimizer='adam', loss='mae')
    elif mname == 'mlp' and itype == 'sequential':
        model = SequentialMLP(in_steps, in_features, 
                               out_features, pred_activation=pred_activation,
                               **kwargs)
    elif mname == 'mlp-diff' and itype == 'supervised':
        model = SupervisedDiffMLP(in_steps, in_features, 
                               out_features, pred_activation=pred_activation,
                               **kwargs)    
        model.compile(optimizer='adam', loss='mae')
    elif mname == 'mlp-diff' and itype == 'sequential':
        model = SequentialDiffMLP(in_steps, in_features, 
                               out_features, pred_activation=pred_activation,
                               **kwargs)
    else:
        raise NotImplementedError
    
    model.compile(optimizer='adam', loss='mae')

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
    plt.title(f"{mname}, {itype}, {sname}, {dname}, "\
              f"{train_val_params['overlap']}, "\
              f"{train_val_params['shuffle']}")
    plt.savefig(OUT_PATH / f"fit.png")

    # Save model

    model.save(OUT_PATH / 'model', save_format='tf')
        
    # Save traininig / validation data for later evaluation
    
    data = {'train_X': train_X, 'train_y': train_y, 
            'val_X': test_X, 'val_y': test_y, 
            'train_indices_y': train_indices_y, 
            'val_indices_y': test_indices_y}
    np.savez(OUT_PATH / f'train_val_data', **data)
    
    # Create and save output config
    
    model_config = {
        'dataset name': dname,
        'dataset shape': dataset.shape,
        'dataset original shape': dataset_original.shape,
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
        'train overlap': train_val_params['overlap'],
        'train shuffle': train_val_params['shuffle'],
        'fit shuffle': FIT_SHUFFLE,
        **kwargs
    }
    
    json.dump(model_config, open(OUT_PATH / 'model_config.json', 'w'))

if __name__ == '__main__':
    main()