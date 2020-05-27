#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers import Input
from keras.models import Model

import plotly.express as px
from tqdm.keras import TqdmCallback

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


from utils.read_dataset import read_dataset
from utils.detrend import detrend
from utils.make_samples import make_samples

# Read dataset
df, districts = read_dataset()
numnans = 0

for i, district in enumerate(districts):
    if i < 53:
        continue
    print(f"District {i}/469: ", district)
    ## Make training samples
    df_district = df[df['district']==district]
    features = df_district.iloc[:, np.r_[7:35]]
    length = features.shape[0]

    if length < 9:
        continue

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
    
    print("SHAPE", samples.shape, features.shape, outputs.shape)
    print("NDIMS", outputs.ndim)
    if outputs.ndim == 1:
        continue
    
    ### No train test split
    train_size = int(len(samples) * 1)
    test_size = len(samples) - train_size
    X_train, X_test = samples[0:train_size,:], samples[train_size:len(samples),:]
    y_train, y_test = outputs[0:train_size,:], outputs[train_size:len(samples),:]

    # # OUTPUTS ARE 3D!!!!!!!!!
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

    #### Make decoder layer inputs
    X_decode_train = X_train[:, 3:, :]
    X_decode_test = X_test[:, 3:, :]

    # Seq 2 Seq Model
    latent_dim = 32

    encoder_inputs = Input(shape=(window_size,X_train.shape[2]))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(output_window_size,X_test.shape[2]))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(1)
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit([X_train, X_decode_train], y_train,
            batch_size=4,
            epochs=100,
            validation_split=0)
        
    X_pred = test_samples
    X_decode_pred = test_samples[:, 3:, :]
    orig_output = model.predict([X_pred,X_decode_pred])
    if np.isnan(orig_output).any():
        print(f"FOUND NAN for {district}")
        numnans += 1
    else:    
        # SAVE THE MODEL
        model.save(f"models/{district}.h5")
        print(district, file=open('train_districts.csv', 'w+'))

print("Nans: ", numnans)