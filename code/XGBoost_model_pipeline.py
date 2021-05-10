#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:31:00 2021

This class is a pipeline for using DNN models for mode analysis. It would preprocess the csv files, train a model, analyse it, and output the economic metrics to files.
You can run only some steps

@author: Huanfa chen
"""

# from analysis import Analysis
# from typing_extensions import TypeVarTuple
# from Analysis_Neural_Net import Analysis_Neural_Net
from Analysis_XGBoost import Analysis_XGBoost
from Plot_Neural_Net import Plot_Neural_Net
from Preprocess_data_DNN import Preprocess_data_DNN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import sys
import os
# from dnn import FeedForward_DNN
from XGBoost_Model import XGBoost_Model

if __name__ == "__main__":
       print("This is the main function...")
       ## parameters of the workflow
       # if PROCESS_RAW_DATA is True, the original data (.csv) will be processed and saved as two .pkl files. Otherwise, the processed files will be used (you need to guarantee that the .pkl files have be generated)
       PROCESS_RAW_DATA = False
       # if RETRAIN_MODEL is True, DNN models will be retrained and saved as tf and .pkl files. Otherwise, the following steps will use the files saved from previous trainings.
       RETRAIN_MODEL = False
       # if ANALYSIS_MODEL is True, the DNN models will be analysed. Otherwise, the result files saved from previous analysis will be used for plotting
       ANALYSIS_MODEL = True
       # if OUTPUT_FILES is True, the trained DNN models will be analysed and .csv files will be saved. It is ignored if ANALYSIS_MODEL is False
       OUTPUT_FILES = True
       # if OUTPUT_PLOTS is True, the trained DNN models will be analysed and plots will be saved
       OUTPUT_PLOTS = True

       # if True, use the designated testing data. Otherwise, use [Data]_X_Test.csv and [Data]_Y_Test.csv
       DESIGNATED_TEST_DATA = True
       file_X_test = 'Original_X_Test.csv'
       file_Y_test = 'Original_Y_Test.csv'

       ## number of models. For quick testing, you can set num_models = 1
       num_models = 5

       # list of dataset and method names. Should have the same length
       list_data_name = ['Original']
       list_method_name = ['xgb']
       for data_name, method_name in zip(list_data_name, list_method_name):
              # variables in London Dataset (14 vars)
              variables = ['age', 'male', 'driving_license',
                     'car_ownership', 'distance', 'dur_walking', 'dur_cycling',
                     'dur_pt_access', 'dur_pt_inv', 'dur_pt_int_total',
                     'pt_n_interchanges', 'dur_driving', 'cost_transit',
                     'cost_driving_total']

              # standard_vars stands for continuous variables  (11 vars)
              standard_vars = ['age', 'distance', 'dur_walking', 'dur_cycling',
                     'dur_pt_access', 'dur_pt_inv', 'dur_pt_int_total',
                     'pt_n_interchanges', 'dur_driving', 'cost_transit',
                     'cost_driving_total']

              # the index of continuous variables in 'variables'
              standard_vars_idx = list(map(lambda x:variables.index(x), standard_vars))
              # [0,4,5,6,7,8,9,10,11,12,13]
              drive_cost_idx = variables.index('cost_driving_total')
              drive_cost_idx_standard = standard_vars.index('cost_driving_total')
              drive_time_idx = variables.index('dur_driving')
              pt_cost_idx = variables.index('cost_transit')
              pt_cost_idx_standard = standard_vars.index('cost_transit')
              pt_time_idx = variables.index('dur_pt_inv')

              modes = ['walk', 'pt', 'cycle', 'drive']
              drive_mode_idx = modes.index('drive')
              pt_mode_idx = modes.index('pt')

              # data_dir = 'data/london/london_processed.pkl'
              # raw_data_dir =  'data/london/london_processed_raw.pkl'
              suffix = ''

              # number of alternative modes
              num_alt = len(modes)
              INCLUDE_VAL_SET = False
              INCLUDE_RAW_SET = True
              
              # N_bootstrap_sample = None
              
              num_training_samples = 7000
              # df = []
              
              # the current wd is 'dnn-for-economic-information'
              dir_data = 'Data'
              dir_model = 'Models'
              dir_result = 'Results'
              dir_plots = 'Plots'

              # path to the csv data files (training and testing)
              path_X_train_csv = os.path.join(dir_data, "{}_X_Train.csv".format(data_name))
              path_Y_train_csv = os.path.join(dir_data, "{}_Y_Train.csv".format(data_name))
              if DESIGNATED_TEST_DATA is True:
                     path_X_test_csv = os.path.join(dir_data, file_X_test)
                     path_Y_test_csv = os.path.join(dir_data, file_Y_test)
              else:
                     path_X_test_csv = os.path.join(dir_data, "{}_X_test.csv".format(data_name))
                     path_Y_test_csv = os.path.join(dir_data, "{}_Y_test.csv".format(data_name))

              # path to the raw and processed data (pkl)
              path_data_raw_pkl = os.path.join(dir_data, "{}_raw.pkl".format(data_name))
              path_data_processed_pkl = os.path.join(dir_data, "{}_processed.pkl".format(data_name))
              # path to the model pickle file, which saves a Analysis_Neural_Net object
              path_model_pickle = os.path.join(dir_model, "{}_{}_Model.pkl".format(data_name, method_name))

              print("Checking folder")
              print(os.path.exists(dir_model))

              if PROCESS_RAW_DATA is True:
                     Preprocess_data_DNN(path_X_train_csv, path_Y_train_csv, path_X_test_csv, path_Y_test_csv, path_data_raw_pkl, path_data_processed_pkl, variables, standard_vars)

              if RETRAIN_MODEL is True:

                     # loading raw and processed data       
                     with open(path_data_processed_pkl, 'rb') as f:
                            input_data = pickle.load(f)
                     with open(path_data_raw_pkl, 'rb') as f:
                            input_data_raw = pickle.load(f)
                     # input_data = pickle.load(f)
                     # train
                     for i in range(num_models):
                            ## TODO: implement training code for XGBoost
                            
                            model_tf_file_name = "{}_{}_model_{}".format(data_name, method_name, str(i))
                            # F_DNN = FeedForward_DNN(num_alt,model_tf_file_name,INCLUDE_VAL_SET,INCLUDE_RAW_SET, dir_model)
                            xgb = XGBoost_Model(num_alt,model_tf_file_name,INCLUDE_VAL_SET,INCLUDE_RAW_SET, dir_model)
                            xgb.load_data(input_data, input_data_raw, num_training_samples)
                            xgb.init_hyperparameter_space() # could change the hyperparameter here by using F_DNN.change_hyperparameter(new)
                            # with rand as False, F_DNN will use the optimal parameter obtained from the hyperparameter tuning. See dnn.py
                            xgb.init_hyperparameter(rand=False) # could change the hyperparameter here by using F_DNN.change_hyperparameter(new)                     
                            xgb.bootstrap_data(N_bootstrap_sample=len(input_data['X_train']))
                            xgb.build_model()
                            # train and save the model
                            xgb.train_model()
              
              if ANALYSIS_MODEL is True:
                     ## TODO: implement training code for XGBoost
                     # why is raw_data_dir necessary here?
                     m = Analysis_XGBoost(dir_model, num_models, variables, standard_vars_idx, modes, path_data_processed_pkl, data_name, method_name, suffix=suffix)
                     m.preprocess(path_data_raw_pkl)
                     m.load_models_and_calculate(mkt=True, disagg=True, social_welfare=True,drive_time_idx=drive_time_idx, drive_cost_idx=drive_cost_idx, drive_cost_idx_standard=drive_cost_idx_standard, \
                                                 drive_time_name='dur_driving', drive_cost_name='cost_driving_total', drive_idx=drive_mode_idx, \
                                                 pt_time_idx=pt_time_idx, pt_cost_idx=pt_cost_idx, pt_idx=pt_mode_idx, time_correction = 1)
                     if OUTPUT_FILES is True:
                            m.write_result(result_dir=dir_result, mean_result_file="{}_{}_mean.csv".format(data_name, method_name), \
                                   model_result_file_prefix="{}_{}_".format(data_name, method_name), model_result_file_suffix='.csv')
                            m.pickle_model(path_model_pickle)
              
              if OUTPUT_PLOTS is True:
                     ## TODO: implement training code for XGBoost

                     # run_dir = 'ldn_models/model15_7000'
                     # run_dir = 'sgp_models/model21'
                     plt.rcParams.update({'font.size': 24})
                     #currency = "£"
                     currency = "$"              

                     '''
                     plot_variables = ['bus_cost ($)', 'bus_ivt (min)', 'rideshare_cost ($)', 'rideshare_ivt (min)', \
                                   'av_cost ($)', 'av_ivt (min)', 'drive_cost ($)', 'drive_ivt (min)']
                     plot_vars = [1, 4, 5, 7, 8, 10, 11, 13]
                     '''
                     # lables of the independent variables in plots
                     plot_variables = ['drive_cost ($)']
                     # index of the independent variables in all_variables
                     plot_vars_standard = [10] # index of standard vars
                     # index of the independent variables in standard vars (aka continuous variables)
                     plot_vars = [13]
                     # name of modes for investigation
                     plot_modes = ['drive']

                     assert len(plot_variables) == len(plot_vars) == len(plot_vars_standard) == len(plot_modes)
                     plot_NN = Plot_Neural_Net(path_model_pickle, dir_plots, data_name, method_name)
                     plot_NN.plot_vot('drive', currency)
                     plot_NN.plot_vot('pt', currency)
                     
                     plot_NN.plot_choice_prob(plot_modes, plot_vars, plot_vars_standard, plot_variables)
                     plot_NN.plot_prob_derivative(plot_modes, plot_vars, plot_vars_standard, plot_variables)#, [86,17,31], ['C1: 0.602', 'C2: 0.595', 'C3: 0.586'])
                     plot_NN.plot_substitute_pattern(plot_vars, plot_vars_standard, plot_variables)


       # model training
       # model analysis
       # writing the results
       # plots

