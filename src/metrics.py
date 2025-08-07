import numpy as np

def rmse(pred, true):
    pred = np.asarray(pred); true = np.asarray(true)
    return float(np.sqrt(np.mean((pred - true) ** 2)))

def mae(pred, true):
    pred = np.asarray(pred); true = np.asarray(true)
    return float(np.mean(np.abs(pred - true)))

def r2(pred, true):
    pred = np.asarray(pred); true = np.asarray(true)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
