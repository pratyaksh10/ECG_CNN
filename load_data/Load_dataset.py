import wfdb
import pandas as pd
import numpy as np
from os import listdir
import os

os.chdir(r'C:\Users\praop\Desktop\Research\ECG_FEATURE_EXTRACTION\DATASET')

file_path = 'mit-bih-arrhythmia-database-1.0.0/'

patients = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111',
            '112', '113', '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
            '200', '201', '202', '203', '205', '207', '208', '209', '210','212', '213','214','215',
            '217', '219', '220', '221', '222', '223','228', '230', '231', '232','233','234']

data = pd.DataFrame()

# EXTRACT ECG FROM FILE

for p in patients:

    file = file_path + p
    ann = wfdb.rdann(file, 'atr')
    sym = ann.symbol

    val, count = np.unique(sym, return_counts=True)
    df_temp = pd.DataFrame({'sym': val, 'val':count, 'p':[p]*len(count)})

    data = pd.concat([data,df_temp],axis=0)


abnormal = ['L', 'R', 'V', '/', 'A', 'f' , 'F', 'j', 'a', 'E', 'J', 'e', 'S']

data['classification'] = -1
data.loc[data.sym == 'N' , 'classification'] = 0
data.loc[data.sym.isin(abnormal), 'classification'] = 1


def extract_ecg(file_path):

    record = wfdb.rdrecord(file_path)

    ann = wfdb.rdann(file_path,'atr')

    signal = record.p_signal

    atr_symbol = ann.symbol
    atr_sample = ann.sample

    return signal, atr_symbol, atr_sample


def create_dataset(pts, window, frequency, abnormal):


    n_columns  = 2 * window * frequency
    X_total = np.zeros((1, n_columns ))
    Y_total = np.zeros((1, 1))
    sym_total = []

    # Keep track of beats across rows
    row_counts = []

    for pt in pts:
        file = file_path + pt

        p_signal, atr_symbol, atr_sample = extract_ecg(file)

        # select first signal
        p_signal = p_signal[:, 0]

        # Non-beats discarded
        df_ann = pd.DataFrame({'atr_symbol': atr_symbol,
                               'atr_sample': atr_sample})
        df_ann = df_ann.loc[df_ann.atr_symbol.isin(abnormal + ['N'])]

        X, Y, sym = build_XY(p_signal, df_ann, n_columns , abnormal)
        sym_total = sym_total + sym
        row_counts.append(X.shape[0])
        X_total = np.append(X_total, X, axis=0)
        Y_total = np.append(Y_total, Y, axis=0)

    # first row dropped
    X_total = X_total[1:, :]
    Y_total = Y_total[1:, :]


    return X_total, Y_total, sym_total


def build_XY(p_signal, df_ann, n_columns , abnormal):    # Building X and Y for each beat



    num_rows = len(df_ann)

    X = np.zeros((num_rows, n_columns ))
    Y = np.zeros((num_rows, 1))
    sym = []

    # keep track of rows
    row_count = 0

    for atr_sample, atr_symbol in zip(df_ann.atr_sample.values, df_ann.atr_symbol.values):

        left = max([0, (atr_sample - window * frequency)])
        right = min([len(p_signal), (atr_sample + window * frequency)])
        x = p_signal[left: right]
        if len(x) == n_columns :
            X[row_count, :] = x
            Y[row_count, :] = int(atr_symbol in abnormal)
            sym.append(atr_symbol)
            row_count += 1
    X = X[:row_count, :]
    Y = Y[:row_count, :]
    return X, Y, sym

window = 3
frequency = 360


X_total, Y_total, sym_total = create_dataset(patients, window, frequency, abnormal)

