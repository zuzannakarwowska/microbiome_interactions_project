import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from utils.measures import (calculate_f1score, calculate_spearman, 
                            calculate_nrmse, inter_dissimilarity, 
                            intra_dissimilarity)


def calculate_measures(true, pred):
    true = pd.DataFrame(true)
    pred = pd.DataFrame(pred)
    names = ["f1score", "spearman", "nrmse", "inter", "intra"]
    f1score = calculate_f1score(true, pred, return_tuple=True)
    spearman = calculate_spearman(true, pred, return_tuple=True)
    nrmse = calculate_nrmse(true, pred, return_tuple=True)
    inter = inter_dissimilarity(true, pred, return_tuple=True)
    _, _, _, _, intra = intra_dissimilarity(true, pred)
    return dict(zip(names, (f1score, spearman, nrmse, inter, intra)))


def plot_series(train_true, train_pred, train_naive, 
                test_true, test_pred, test_naive, group):
    if train_true is not None and train_pred is not None:
        plt.figure(figsize=(15,5))
        plt.subplot(1, 2, 1)
        plt.plot(train_true[group-1], label='true')
        plt.plot(train_pred[group-1], label='predicted')
        if train_naive is not None:
            plt.plot(train_naive[group-1], label='naive')
        # plt.plot(abs(test_inv_yhat_super[group-1] - \
        #              test_inv_y[group-1]))  # absolute difference
        rmse = np.sqrt(mean_squared_error(train_true[group-1], 
                                          train_pred[group-1]))
        plt.title(f'Train (group: {group}, RMSE: {rmse:.2f})')
        plt.legend()
        plt.subplot(1, 2, 2)
    plt.plot(test_true[group-1], label='true')
    plt.plot(test_pred[group-1], label='predicted')
    if train_naive is not None:
        plt.plot(test_naive[group-1], label='naive')
    # plt.plot(abs(test_inv_yhat_super[group-1] - \
    #              test_inv_y[group-1]))  # absolute difference
    rmse = np.sqrt(mean_squared_error(test_true[group-1],
                                      test_pred[group-1]))
    plt.title(f'Test (group: {group}, RMSE: {rmse:.2f})')
    plt.legend()
    plt.show()
    
    
if __name__ == '__main__':
    pass