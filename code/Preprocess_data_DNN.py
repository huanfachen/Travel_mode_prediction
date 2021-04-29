# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:31:00 2021

The function of Preprocess_data_DNN is used to preprocess the CSV data and save them as pickle files.

@author: Huanfa chen
"""

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def Preprocess_data_DNN (file_X_train, file_Y_train, file_X_test, file_Y_test, file_pkl_raw, file_pkl_processed, variables, cont_variables):
       """Preprocess the csv files and output pickle files

       Args:
           file_X_train (str): path to X train data
           file_Y_train (str): path to Y train data
           file_X_test (str): path to X test data
           file_Y_test (str): path to Y test data
           file_pkl_raw (str): path to the pickle file of raw data
           file_pkl_processed (str): path to the pickle file of processed data
           variables (list of str): list of variables
           cont_variables (list of str): list of continuous variables
       """       

       X_train = pd.read_csv(file_X_train)[variables]
       Y_train = pd.read_csv(file_Y_train)
       X_test = pd.read_csv(file_X_test)[variables]
       Y_test = pd.read_csv(file_Y_test)

       choice_map = {'walk':0, 'pt':1, 'cycle':2, 'drive':3}
       Y_train = Y_train['travel_mode'].map(choice_map)
       Y_test = Y_test['travel_mode'].map(choice_map)

       input_data = {}
       input_data['X_train'] = X_train
       input_data['X_test'] = X_test
       input_data['Y_train'] = Y_train
       input_data['Y_test'] = Y_test

       with open(file_pkl_raw, 'wb') as f:
              pkl.dump(input_data, f)

       # standardisation
       scaler = StandardScaler().fit(input_data['X_train'][variables])
       X_train = scaler.transform(input_data['X_train'][variables])
       X_test = scaler.transform(input_data['X_test'][variables])
       
       input_data = {}
       input_data['X_train'] = pd.DataFrame(X_train, columns=variables)
       input_data['X_test'] = pd.DataFrame(X_test, columns=variables)
       input_data['Y_train'] = Y_train
       input_data['Y_test'] = Y_test
       # data = pd.DataFrame(StandardScaler().fit_transform(data[variables]), columns = variables)
       # X_train, X_test, y_train, y_test = train_test_split(data, choice, test_size=0.10, random_state=42)
       
       # output
       with open(file_pkl_processed, 'wb') as f:
              pkl.dump(input_data, f)
              pkl.dump(variables, f)
              pkl.dump(cont_variables, f)

if __name__ == "__main__":
       None