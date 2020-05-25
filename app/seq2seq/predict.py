#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import keras
from keras.models import load_model

import plotly.express as px
from tqdm.keras import TqdmCallback

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


from .utils.read_dataset import read_dataset
from .utils.detrend import detrend
from .utils.make_samples import make_samples


def get_predictions(district):
    # Read dataset
    df, districts = read_dataset()

    district = district.replace(' ', '')

    ## Make training samples
    # district = 'bengaluruurban'
    df_district = df[df['district']==district]
    features = df_district.iloc[:, np.r_[7:35]]
    length = features.shape[0]

    y_r = np.array(df_district.iloc[:, 2].values)
    y_c = df_district.iloc[:, 3].values
    y_d = df_district.iloc[:, 4].values
    y_a = df_district.iloc[:, 5].values

    detrend_window_size = 3
    detrended_r, trend_r = detrend(y_r[:], detrend_window_size)
    detrended_c, trend_c = detrend(y_c[:], detrend_window_size)
    detrended_d, trend_d = detrend(y_d[:], detrend_window_size)
    detrended_a, trend_a = detrend(y_a[:], detrend_window_size)



    ## Normalize Detrended data
    if np.mean(y_r) != 0:
        detrended_r /= np.mean(y_r)
    if np.mean(y_c) != 0:
        detrended_c /= np.mean(y_c)
    if np.mean(y_d) != 0:
        detrended_d /= np.mean(y_d)
    if np.mean(y_a) == 0:
        detrended_a /= np.mean(y_a)

    # Make Samples
    window_size = 6
    output_window_size = 3
    samples, outputs, test_samples = make_samples(window_size, output_window_size, length, features, detrended_r, detrended_c, detrended_d, detrended_a)

    #
    # ### No train test split
    # train_size = int(len(samples) * 1)
    # test_size = len(samples) - train_size
    # X_train, X_test = samples[0:train_size,:], samples[train_size:len(samples),:]
    # y_train, y_test = outputs[0:train_size,:], outputs[train_size:len(samples),:]
    #
    # # # OUTPUTS ARE 3D!!!!!!!!!
    # y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    # y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
    #
    # #### Make decoder layer inputs
    # X_decode_train = X_train[:, 3:, :]
    # X_decode_test = X_test[:, 3:, :]


    # Load Seq 2 Seq Model
    model = load_model(f'app/seq2seq/models/{district}.h5')

    # PREDICTIONS
    N = 4 # Predicts for N * 3 days, 12 days xP
    last_index = y_c.shape[0] - 1
    predict_indices = np.array([24, 25, 26])
    predictions = []
    for i in range(N):
        X_pred = test_samples
        X_decode_pred = test_samples[:, 3:, :]
        orig_output = model.predict([X_pred,X_decode_pred])

        # nake new sample to predict next 3 days
        new_ts = np.copy(test_samples[0])
        new_ts[:3, :] = new_ts[3:, :]
        new_output = orig_output.reshape(3, 1)
        new_ts[-3, 29] = new_output[0]
        new_ts[-2, 29] = new_output[1]
        new_ts[-1, 29] = new_output[2]

        ### Denormalize
        output = orig_output
        output *= np.mean(y_c)
        output = output.reshape(1, 3)

        ### Find trend and add to detrended output
        m1 = (trend_c[-1] - trend_c[-3]) / 2
        b = trend_c[-1] - (last_index * m1)

        ### Find translated trends
        trend_output = m1 * predict_indices + b
        trend_output #

        ### Add trends to outputs
        newoutput = output + trend_output
        predictions.append(newoutput)
        # print(newoutput)

        predict_indices += 3
        test_samples = np.array([new_ts])

    predictions = np.array([predictions])
    predictions = predictions.reshape(1, -1)
    print(predictions)
    return predictions
