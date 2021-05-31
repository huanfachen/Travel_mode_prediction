#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thr May 06 17:01:00 2021

This class is a pipeline for using XGBoost models for mode analysis. It would preprocess the csv files, train a model, analyse it, and output the economic metrics to files.
You can run the whole pipeline or only some steps.

@author: Huanfa chen
"""

# from analysis import Analysis
# from typing_extensions import TypeVarTuple
from Analysis_Neural_Net import Analysis_Neural_Net
from Plot_Neural_Net import Plot_Neural_Net
from Preprocess_data_DNN import Preprocess_data_DNN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import sys
import os
from dnn import FeedForward_DNN

if __name__ == "__main__":
       print("This is the main function...")
       ## parameters of the workflow
       # if PROCESS_RAW_DATA is True, the original data (.csv) will be processed and saved as two .pkl files. Otherwise, the processed files will be used (you need to guarantee that the .pkl files have be generated)
       PROCESS_RAW_DATA = False
       # if RETRAIN_MODEL is True, DNN models will be retrained and saved as tf and .pkl files. Otherwise, the following steps will use the files saved from previous trainings.
       RETRAIN_MODEL = True
       # if ANALYSIS_MODEL is True, the DNN models will be analysed. Otherwise, the result files saved from previous analysis will be used for plotting
       ANALYSIS_MODEL = True
       # if OUTPUT_FILES is True, the trained DNN models will be analysed and .csv files will be saved. It is ignored if ANALYSIS_MODEL is False
       OUTPUT_FILES = True
       # if OUTPUT_PLOTS is True, the trained DNN models will be analysed and plots will be saved
       OUTPUT_PLOTS = False

       # if True, use the designated testing data. Otherwise, use [Data]_X_Test.csv and [Data]_Y_Test.csv
       DESIGNATED_TEST_DATA = True
       file_X_test = 'Original_X_Test.csv'
       file_Y_test = 'Original_Y_Test.csv'

       ## number of models. For quick testing, you can set num_models = 1
       num_models = 10

       # variables in London Dataset
       variables = ['age', 'male', 'driving_license',
              'car_ownership', 'distance', 'dur_walking', 'dur_cycling',
              'dur_pt_access', 'dur_pt_inv', 'dur_pt_int_total',
              'pt_n_interchanges', 'dur_driving', 'cost_transit',
              'cost_driving_total']

       # standard_vars stands for continuous variables 
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

       modes = ['walk', 'cycle', 'pt', 'drive']
       # mapping from travel modes to numbers. Would be used in data preprocessing
       # 'walk':0, 'cycle':1, 'pt':2, 'drive':3
       choice_map = {mode:i for i,mode in enumerate(modes)}

       drive_mode_idx = modes.index('drive')
       pt_mode_idx = modes.index('pt')

       # data_dir = 'data/london/london_processed.pkl'
       # raw_data_dir =  'data/london/london_processed_raw.pkl'
       suffix = ''

       num_alt = len(modes)
       INCLUDE_VAL_SET = False
       INCLUDE_RAW_SET = True
       
       # N_bootstrap_sample = None
       
       num_training_samples = 7000
       # df = []
       
       # the current wd is 'dnn-for-economic-information'
       dir_data = 'data'
       dir_model = 'Models'
       dir_result = 'Results'
       dir_plots = 'Plots'

       # list of dataset and method names
       # In sum, there are 61 datasets (including Original)
       list_all_data = ['Original', '1.1_RandomUnderSampler_10',
       '1.1_RandomUnderSampler_1','1.1_RandomUnderSampler_2','1.1_RandomUnderSampler_3','1.1_RandomUnderSampler_4','1.1_RandomUnderSampler_5','1.1_RandomUnderSampler_6','1.1_RandomUnderSampler_7','1.1_RandomUnderSampler_8','1.1_RandomUnderSampler_9','1.2_One-SidedSelection_10','1.2_One-SidedSelection_1','1.2_One-SidedSelection_2','1.2_One-SidedSelection_3','1.2_One-SidedSelection_4','1.2_One-SidedSelection_5','1.2_One-SidedSelection_6','1.2_One-SidedSelection_7','1.2_One-SidedSelection_8','1.2_One-SidedSelection_9','1.3_NeighbourhoodCleaningRule_10','1.3_NeighbourhoodCleaningRule_1','1.3_NeighbourhoodCleaningRule_2','1.3_NeighbourhoodCleaningRule_3','1.3_NeighbourhoodCleaningRule_4','1.3_NeighbourhoodCleaningRule_5','1.3_NeighbourhoodCleaningRule_6','1.3_NeighbourhoodCleaningRule_7','1.3_NeighbourhoodCleaningRule_8','1.3_NeighbourhoodCleaningRule_9','2.1_RandomOverSampler_10','2.1_RandomOverSampler_1','2.1_RandomOverSampler_2','2.1_RandomOverSampler_3','2.1_RandomOverSampler_4','2.1_RandomOverSampler_5','2.1_RandomOverSampler_6','2.1_RandomOverSampler_7','2.1_RandomOverSampler_8','2.1_RandomOverSampler_9','2.2_SMOTENC_10','2.2_SMOTENC_1','2.2_SMOTENC_2','2.2_SMOTENC_3','2.2_SMOTENC_4','2.2_SMOTENC_5','2.2_SMOTENC_6','2.2_SMOTENC_7','2.2_SMOTENC_8','2.2_SMOTENC_9','2.3_ADASYN_10','2.3_ADASYN_1','2.3_ADASYN_2','2.3_ADASYN_3','2.3_ADASYN_4','2.3_ADASYN_5','2.3_ADASYN_6','2.3_ADASYN_7','2.3_ADASYN_8','2.3_ADASYN_9']
       # list_data_name = list_all_data       
       
       list_second_part = ['2.2_SMOTENC_10','2.2_SMOTENC_1','2.2_SMOTENC_2','2.2_SMOTENC_3','2.2_SMOTENC_4','2.2_SMOTENC_5','2.2_SMOTENC_6','2.2_SMOTENC_7','2.2_SMOTENC_8','2.2_SMOTENC_9','2.3_ADASYN_10','2.3_ADASYN_1','2.3_ADASYN_2','2.3_ADASYN_3','2.3_ADASYN_4','2.3_ADASYN_5','2.3_ADASYN_6','2.3_ADASYN_7','2.3_ADASYN_8','2.3_ADASYN_9']
       list_data_name = list_all_data
       # list_data_name = ['1.1_RandomUnderSampler_10','1.1_RandomUnderSampler_1','1.1_RandomUnderSampler_2','1.1_RandomUnderSampler_3','1.1_RandomUnderSampler_4','1.1_RandomUnderSampler_5','1.1_RandomUnderSampler_6','1.1_RandomUnderSampler_7','1.1_RandomUnderSampler_8','1.1_RandomUnderSampler_9']
       # list_data_name = ['Original']
       list_method_name = ['dnn' for x in list_data_name]
       for data_name, method_name in zip(list_data_name, list_method_name):
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
                     Preprocess_data_DNN(path_X_train_csv, path_Y_train_csv, path_X_test_csv, path_Y_test_csv, path_data_raw_pkl, path_data_processed_pkl, variables, standard_vars, choice_map)

              if RETRAIN_MODEL is True:

                     # loading raw and processed data       
                     with open(path_data_processed_pkl, 'rb') as f:
                            input_data = pickle.load(f)
                     with open(path_data_raw_pkl, 'rb') as f:
                            input_data_raw = pickle.load(f)
                     # input_data = pickle.load(f)
                     # train
                     for i in range(num_models):
                            model_tf_file_name = "{}_{}_model_{}".format(data_name, method_name, str(i))
                            F_DNN = FeedForward_DNN(num_alt,model_tf_file_name,INCLUDE_VAL_SET,INCLUDE_RAW_SET, dir_model)
                            F_DNN.load_data(input_data, input_data_raw, num_training_samples)
                            F_DNN.init_hyperparameter_space() # could change the hyperparameter here by using F_DNN.change_hyperparameter(new)
                            # with rand as False, F_DNN will use the optimal parameter obtained from the hyperparameter tuning. See dnn.py
                            F_DNN.init_hyperparameter(rand=False) # could change the hyperparameter here by using F_DNN.change_hyperparameter(new)                     
                            F_DNN.bootstrap_data(N_bootstrap_sample=len(input_data['X_train']))
                            F_DNN.build_model()
                            # train and save the model
                            F_DNN.train_model()
              
              if ANALYSIS_MODEL is True:
                     # why is raw_data_dir necessary here?
                     m = Analysis_Neural_Net(dir_model, num_models, variables, standard_vars_idx, modes, path_data_processed_pkl, data_name, method_name, suffix=suffix)
                     m.preprocess(path_data_raw_pkl)
                     m.load_models_and_calculate(mkt=True, disagg=True, social_welfare=True,drive_time_idx=drive_time_idx, drive_cost_idx=drive_cost_idx, drive_cost_idx_standard=drive_cost_idx_standard, \
                                                 drive_time_name='dur_driving', drive_cost_name='cost_driving_total', drive_idx=drive_mode_idx, \
                                                 pt_time_idx=pt_time_idx, pt_cost_idx=pt_cost_idx, pt_idx=pt_mode_idx, time_correction = 1)
                     if OUTPUT_FILES is True:
                            m.write_result(result_dir=dir_result, mean_result_file="{}_{}_mean.csv".format(data_name, method_name), \
                                   model_result_file_prefix="{}_{}_".format(data_name, method_name), model_result_file_suffix='.csv')
                            m.pickle_model(path_model_pickle)
              
              if OUTPUT_PLOTS is True:
                     run_dir = 'ldn_models/model15_7000'
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


