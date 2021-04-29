# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 00:16:16 2019

@author: wangqi44
"""

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data = pd.read_csv('data/london/dataset.csv')
# # choice set 01234 -> [walk,bus,ridesharing,drive,av]
# choice_map = {'walk':0, 'pt':1, 'cycle':2, 'drive':3}
# choice = data['travel_mode'].map(choice_map)
# data['dur_pt_inv'] = data['dur_pt_bus'] + data['dur_pt_rail']
# data['male'] = 1-data['female']

# variables = ['age', 'male', 'driving_license',
#        'car_ownership', 'distance', 'dur_walking', 'dur_cycling',
#        'dur_pt_access', 'dur_pt_inv', 'dur_pt_int_total', 
#        'pt_n_interchanges', 'dur_driving', 'cost_transit',
#        'cost_driving_total']
# #       'cost_driving_fuel', 'cost_driving_con_charge']

# standard_vars = ['age', 'distance', 'dur_walking', 'dur_cycling',
#        'dur_pt_access', 'dur_pt_inv', 'dur_pt_int_total', 
#        'pt_n_interchanges', 'dur_driving', 'cost_transit',
#        'cost_driving_total']

# # Export raw data
# X_train, X_test, y_train, y_test = train_test_split(data[variables], choice, test_size=0.10, random_state=42)

# input_data = {}
# input_data['X_train'] = X_train
# input_data['X_test'] = X_test
# input_data['Y_train'] = y_train
# input_data['Y_test'] = y_test

# with open('data/london/london_processed_raw.pkl', 'wb') as f:
#     pkl.dump(input_data, f)


# # Export standardized data
# data = pd.DataFrame(StandardScaler().fit_transform(data[variables]), columns = variables)
# X_train, X_test, y_train, y_test = train_test_split(data, choice, test_size=0.10, random_state=42)

# input_data = {}
# input_data['X_train'] = X_train
# input_data['X_test'] = X_test
# input_data['Y_train'] = y_train
# input_data['Y_test'] = y_test

# with open('data/london/london_processed.pkl', 'wb') as f:
#     pkl.dump(input_data, f)
#     pkl.dump(variables, f)
#     pkl.dump(standard_vars, f)

with open("data/london/london_processed_raw.pkl", 'rb') as f:
       data = pkl.load(f)
print(type(data['X_train']))
print(type(data['Y_train']))
print(data['Y_train'])
# save as csv files
# data = pd.read_csv('data/london/dataset.csv')
# X_train, X_test, y_train, y_test = train_test_split(data[variables], choice, test_size=0.10, random_state=42)
# choice = data['travel_mode']
variables = ['age', 'male', 'driving_license',
       'car_ownership', 'distance', 'dur_walking', 'dur_cycling',
       'dur_pt_access', 'dur_pt_inv', 'dur_pt_int_total', 
       'pt_n_interchanges', 'dur_driving', 'cost_transit',
       'cost_driving_total']
pd.DataFrame(data['X_train'], columns=variables).to_csv("Data/London_original_X_train.csv")
data['Y_train'].to_frame(name="mode").to_csv("Data/London_original_Y_train.csv")
pd.DataFrame(data['X_test'], columns=variables).to_csv("Data/London_original_X_test.csv")
data['Y_test'].to_frame(name="mode").to_csv("Data/London_original_Y_test.csv")
# pd.DataFrame(data['Y_test'], columns=["mode"]).to_csv("Data/original_Y_test.csv")
    
