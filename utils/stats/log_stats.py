from loguru import logger
import numpy as np
import datetime 

def getQuantile(arr):
    arr = np.array(arr)
    quantiles_point = [10, 25, 50, 75, 90]
    return [f'{np.quantile(arr, q / 100):.3f}' for q in quantiles_point]

def getQuantileInfo(arr):
    quantiles = getQuantile(arr)
    return f'Quantile 10, 25, 50, 75, 90 = {" ".join(quantiles)}'

def logRecord(outputManager, arr, record_name):
    outputManager.writeLog(f'{record_name}: Mean={arr.mean():.4f}, Max={arr.max():.4f}, Min={arr.min():.4f}, {getQuantileInfo(arr)}')

def writeLog(outputManager, epoch, train_loss, train_r2, test_loss, test_r2):
    train_loss = np.array(train_loss)
    train_r2 = np.array(train_r2)
    test_loss = np.array(test_loss)
    test_r2 = np.array(test_r2)

    outputManager.writeLog(f'Finish training Epoch {epoch}')
    logRecord(outputManager, train_loss, 'Train Loss')
    logRecord(outputManager, train_r2, 'Train R2')
    logRecord(outputManager, test_loss, 'Test Loss')
    logRecord(outputManager, test_r2, 'Test R2')
