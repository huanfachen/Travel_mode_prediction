# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:51:14 2021

@author: Huanfa Chen
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import xgboost

class Analysis_XGBoost:
    """
    Reading in the XGBoost model (in pickle format) and calculating relevant economic metrics
    """
    def __init__(self, run_dir, numModels, all_vars, cont_vars, modes, data_dir, data_name, method_name, suffix=''):
        """
        The NN models from Tensorflow should be saved properly in the run_dir folder. 
        Meta graph files and model files are named as "model[num].ckpt.meta" and "model[num].ckpt", respectively, with "[num]" indicating the number of the model.

        Args:
            run_dir (str): folder path
            numModels (int): number of model runs
            all_vars (list of str): the name list of all variable
            cont_vars (list of str): the index list of continuous variables in self.all_variables
            modes (list of str): the list of travel modes
            data_dir (str): the path to the processed data file (.pkl).
            suffix (str, optional): the suffix used to distinguish scenarios. Defaults to ''.
        """        
        self.all_variables = all_vars
        # indexes of X_train_standard
        self.cont_vars = cont_vars
        # names of continuous variables
        self.cont_vars_name = [self.all_variables[x] for x in self.cont_vars]
        # standard deviation of X_train_standard
        self.std = None
        # all variables' raw values
        self.X_train_raw = None
        # all continuous variables' raw values
        self.X_train_standard = None
        self.X_test_standard = None
        self.all_variables = all_vars
        self.modes = modes
        self.run_dir = run_dir
        self.numModels = numModels
        # standardized input feeding into NN
        self.numAlt = len(self.modes)
        self.colors = ['salmon', 'wheat','darkseagreen','plum','dodgerblue']

        self.suffix = suffix
        with open(data_dir, 'rb') as f:
            SGP = pickle.load(f)
        self.input_data = {}
        self.input_data['X_train'] = SGP['X_train'+self.suffix]
        self.input_data['Y_train'] = SGP['Y_train'+self.suffix]
        self.input_data['X_test'] = SGP['X_test'+self.suffix]
        self.input_data['Y_test'] = SGP['Y_test'+self.suffix]

        # filtered_train and filterd_test will be used later to retain the non-NA records
        self.filterd_train = []
        self.filterd_test = []

        numContVars = len(self.cont_vars)
        self.numIndTrn = len(self.input_data['Y_train'])
        self.numIndTest = len(self.input_data['Y_test'])

        # market share
        self.mkt_share_train = np.zeros((self.numModels, self.numAlt))
        self.mkt_share_test = np.zeros((self.numModels, self.numAlt))

        # Metrics below are based on probability derivatives, and are only valid for continuous variables
        # (variable, model #, mode #, individual # in training set)

        self.mkt_prob_derivative = np.zeros((numContVars, self.numModels, self.numAlt, self.numIndTrn))
        self.prob_derivative = np.zeros((numContVars, self.numModels, self.numAlt, self.numIndTrn))
        # mkt: market average individual, the individual dimension has only one variable; the others fixed at market average
        # not mkt: actual individuals with all the variables
        
        self.elasticity = np.zeros((numContVars, self.numModels, self.numAlt, self.numIndTrn))

        # The metrics of probability, utility, and elasticity only work for continuous variables
        self.prob_derivative_test = np.zeros((numContVars, self.numModels, self.numAlt, self.numIndTest))
        self.util_derivative_test = np.zeros((numContVars, self.numModels, self.numIndTest))
        self.elasticity_test = np.zeros((numContVars, self.numModels, self.numAlt, self.numIndTest))
        # (variable, mode, individual #) - average choice prob
        self.average_cp = np.zeros((numContVars, self.numAlt, self.numIndTrn))
        # (variable, model #, mode, individual #) - choice prob
        self.mkt_choice_prob = np.zeros((numContVars, self.numModels, self.numAlt, self.numIndTrn))
        self.vot_drive_train = np.zeros((self.numModels, self.numIndTrn))
        self.vot_drive_test = np.zeros((self.numModels, self.numIndTest))
        self.vot_pt_train = np.zeros((self.numModels, self.numIndTrn))
        self.vot_pt_test = np.zeros((self.numModels, self.numIndTest))

        # model pickle file
        self.list_model_file = []
        
        for index in range(self.numModels):

            # meta_graph_file = os.path.join(self.run_dir, "{}_{}_model_{}.ckpt.meta".format(data_name, method_name, str(index)))
            model_file = os.path.join(self.run_dir, "{}_{}_model_{}.pkl".format(data_name, method_name, str(index)))
            # check if files exist. If not, throw FileNotFoundError
            # try:
            #     file_test = open(model_file, 'r')
            #     # file_test = open(model_file, 'r')
            # except FileNotFoundError as e:
            #     print("Wrong file or file path")
            #     print(e.filename)

            self.list_model_file.append(model_file)

    def compute_XGBoost_gradients(self, input_x, vars, xgb_model):
        """Computing the probability derivative of the var

        Args:
            input_x (pd.DataFrame): data frame of X_train
            vars (list of str): names of variables
            xgb_model (XGBoost classifier): a model of XGBoost classifier

        Returns:
            prob_derivative: an array of the shape (num_var, num_alt, num_instance)
        """        
        input_x = np.array(input_x)
        prob_derivative = []
        temp_prob = xgb_model.predict_proba(input_x)
        for var in vars:
            # get the idx of var in all_variables
            idx = self.all_variables.index(var)
            increment = np.ptp(input_x[:, idx]) / 10
            if increment <= 0:
                increment = 1e-3
            # increase x_feed by increment
            input_x[:, idx] = input_x[:, idx] + increment
            new_prob = xgb_model.predict_proba(input_x)
            gradient = (new_prob - temp_prob) / increment
            prob_derivative.append(np.array(gradient).T)
            # restore input_x
            input_x[:, idx] = input_x[:, idx] - increment
        return np.array(prob_derivative)

    def compute_prob_curve(self, input_x, vars, xgb_model):
        """Computing the probability and probability derivative using a market average person

        Args:
            input_x (pd.DataFrame): data frame of X_train
            vars (list of str): names of continous variables (Not index)
            xgb_model (XGBoost classifier): a model of XGBoost classifier

        Returns:
            [choice_prob, prob_derivative]: two np arrays representing the choice probability and probibility derivative. Both in the shape of (num_var, num_alt, num_instance)
        """        
        input_x = np.array(input_x)
        x_avg = np.mean(input_x, axis=0)
        x_feed = np.repeat(x_avg, len(input_x)).reshape(len(input_x), np.size(input_x, axis=1), order='F')
        choice_prob = []
        prob_derivative = []
        for var in vars:
            # get the idx of var in all_variables
            idx = self.all_variables.index(var)
            x_feed[:, idx] = input_x[:, idx]
            # compute probability
            temp_prob = xgb_model.predict_proba(x_feed)
            choice_prob.append(np.array(temp_prob).T)
            # compute gradient
            # deciding the interval. Should be greater than zero.
            # If the interval is too small, the probability change is insignificant. Tree-based models are insensitive to very small changes in feature values 
            increment = np.ptp(x_feed[:, idx]) / 10
            if increment <= 0:
                increment = 1e-3
            # increase x_feed by increment
            x_feed[:, idx] = x_feed[:, idx] + increment
            new_prob = xgb_model.predict_proba(x_feed)
            gradient = (new_prob - temp_prob) / increment
            prob_derivative.append(np.array(gradient).T)
            # restore x_feed to x_avg
            x_feed[:, idx] = x_avg[idx]
        return np.array(choice_prob), np.array(prob_derivative)

        # input_x = np.array(input_x)
        # x_avg = np.mean(input_x, axis=0)
        # # len(input_x) returns the number of rows of input_x
        # # np.size(input_x, axis=1) returns the number of columns in input_x.
        # # x_feed would have identical rows, with each row being the row average of input_x
        # x_feed = np.repeat(x_avg, len(input_x)).reshape(len(input_x), np.size(input_x, axis=1), order='F')
        # choice_prob = []
        # prob_derivative = []

        # for idx in var:
        #     x_feed[:, idx] = input_x[:, idx]
        #     temp = sess.run(prob, feed_dict={x: x_feed})
        #     choice_prob.append(np.array(temp).T)
        #     temp = []
        #     for j in range(self.numAlt):
        #         grad = tf.gradients(prob[:, j], x)
        #         temp1 = sess.run(grad, feed_dict={x: x_feed})
        #         temp1 = temp1[0][:, idx]
        #         temp.append(temp1)
        #     prob_derivative.append(np.array(temp))
        #     x_feed[:, idx] = x_avg[idx]

        # return np.array(choice_prob), np.array(prob_derivative)

    def load_models_and_calculate(self, mkt, disagg, social_welfare, drive_time_idx, drive_cost_idx, drive_cost_idx_standard, drive_time_name, drive_cost_name, drive_idx, pt_time_idx, pt_cost_idx, pt_idx, time_correction):
        """Load the model, calculate and output the economic information.

        Args:
            mkt (bool): if True, calcualte the choice prob and Prob Derivative for market average person
            disagg (bool): if True, calcualte the disaggregate prob derivative and elasticity. Otherwise, calcualte the average.
            social_welfare (bool): if True, calculate the welfare.
            drive_time_idx (int): the index of the drive time column in self.variables
            drive_cost_idx (int): the index of the drive time column in self.variables
            drive_cost_idx_standard (int): the index of the drive time column in self.standard_vars_idx
            drive_time_name (str): column name of the driving time variable
            drive_cost_name (str): column name of the driving cost variable
            drive_idx (int): the index of driving in self.modes (index starting from 0)
            pt_time_idx (int): the index of the pt time column in self.variables
            pt_cost_idx (int): the index of the pt cost column in self.variables
            pt_idx (int): the index of pt in self.modes (index starting from 0)
            time_correction (int): the correction factor of the driving time unit
        """        
        self.filterd_train = []
        self.filterd_test = []
        sw_ind = []
        
        # prediction
        self.predict_test = []
        
        for index in range(self.numModels):
            # load pickle file
            with open(self.list_model_file[index], 'rb') as f:
                xgb_model = pickle.load(f)
            # xgb_model = pickle.load(self.list_model_file[index])
            # calculate prob_test and prediction_test
            # return: a numpy array of shape array-like of shape (n_samples, n_classes) with the probability of each data example being of a given class.
            prob_test = xgb_model.predict_proba(self.input_data['X_test'], validate_features = False)
            predict_test = xgb_model.predict(self.input_data['X_test'], validate_features = False)
            self.predict_test.append(predict_test)

            # market share
            # self.mkt_share_train[index, :] = np.sum(prob_train, axis = 0) / self.numIndTrn
            self.mkt_share_test[index, :] = np.sum(prob_test, axis = 0) /self.numIndTest
            # print(self.mkt_share_train[index, :])

            if disagg:
                # Disaggregate Prob Derivative and Elasticity
                # grad = []
                drive_time = drive_time_idx
                drive_cost = drive_cost_idx
                # shape of gradients_all_modes: (num_var, num_alt, num_instance)
                gradients_all_modes = self.compute_XGBoost_gradients(self.input_data['X_test'], self.all_variables, xgb_model)
                for j in range(len(self.modes)):
                    # grad.append(tf.gradients(prob[:, j], x))

                    # testing data
                    # shape of gradients: (num_instance, num_var)
                    gradients = gradients_all_modes[:,j,:].T
                    # shape of prob_derivative_test: (numContVars, self.numModels, self.numAlt, self.numIndTest)
                    self.prob_derivative_test[:, index, j, :] = np.array(gradients[:,self.cont_vars]).T
                    elas = gradients[:, self.cont_vars] / prob_test[:, j][:, None] * np.array(self.X_test_standard)[:, self.cont_vars] / self.std[None,self.cont_vars]
                    self.elasticity_test[:, index, j, :] = np.array(elas).T
                    
                    if j == drive_idx: # If drive, then calculate VOT_drive
                        # filter out bad records
                        # print how many observations are filtered out
                        # Value of time : dp/dtime over dp/dcost, correct for normalization as well as units
                        # Take the mean of the trial
                        v = gradients[:, drive_time] / gradients[:, drive_cost] / self.std[None, drive_time] * self.std[None, drive_cost] * time_correction
                        self.vot_drive_test[index,:] = v
                        filt = ~np.isnan(v) & ~np.isinf(v)
                        v = v[filt]
                        print("Model ", index, ": dropped ", self.numIndTest - len(v), " testing observations.")
                        self.filterd_test.append(self.numIndTest - len(v))

                        # util_derivative = tf.gradients(utility[:, j], x)
                        # util_derivative_test = sess.run(util_derivative, feed_dict={x: self.input_data['X_test']})[0][:, self.cont_vars]
                        # self.util_derivative_test[:, index, :] = np.array(util_derivative_test).T

                    if j == pt_idx: # If pt, then calculate VOT_pt
                        # filter out bad records
                        # print how many observations are filtered out
                        # Value of time : dp/dtime over dp/dcost, correct for normalization as well as units
                        # Take the mean of the trial
                        v = gradients[:, pt_time_idx] / gradients[:, pt_cost_idx] / self.std[None, pt_time_idx] * self.std[None, pt_cost_idx] * time_correction
                        self.vot_pt_test[index,:] = v
                        filt = ~np.isnan(v) & ~np.isinf(v)
                        v = v[filt]
                        print("Model ", index, ": dropped ", self.numIndTest - len(v), " testing observations.")
                        self.filterd_test.append(self.numIndTest - len(v))

                        # util_derivative = tf.gradients(utility[:, j], x)
                        # util_derivative_test = sess.run(util_derivative, feed_dict={x: self.input_data['X_test']})[0][:, self.cont_vars]
                        # self.util_derivative_test[:, index, :] = np.array(util_derivative_test).T

            if mkt:
                # Choice prob and Prob Derivative for market average person
                # why using X_train??
                choice_prob, prob_derivative = self.compute_prob_curve(self.input_data['X_train'], self.cont_vars_name, xgb_model)
                # (variable, model #, mode, individual #) - choice prob
                self.mkt_prob_derivative[:, index, :, :] = np.array(prob_derivative)
                self.mkt_choice_prob[:, index, :, :] = np.array(choice_prob)

            if social_welfare:
                None # Not implemented
                # utility0 = sess.run(utility, feed_dict={x: self.input_data['X_test']})
                # new_input = self.input_data['X_test'].copy()
                # new_input[drive_cost_name] += 1
                # utility1 = sess.run(utility, feed_dict={x: new_input})
                # # sw_ind is welfare change per person
                # sw_ind.append(np.log(np.sum(np.exp(utility1), axis=1)) - np.log(np.sum(np.exp(utility0), axis=1)))           
            # calculate gradient

            # delete all below - from DNN

        #     tf.reset_default_graph()

        #     # load the model files (.ckpt.meta) from run_dir. These files should be specified and loaded in the __init__() function.
        #     sess = tf.InteractiveSession()
        #     saver = tf.train.import_meta_graph(self.list_meta_graph_file[index])
        #     saver.restore(sess, self.list_model_file[index])

        #     graph = tf.get_default_graph()

        #     # x denotes the names of the tensors in the first neural layer
        #     x = graph.get_tensor_by_name("X:0")
        #     # x denotes the names of the tensors in the output layer
        #     prob = graph.get_tensor_by_name("prob:0")
        #     prob_train = sess.run(prob, feed_dict={x: self.input_data['X_train']})
        #     prob_test = sess.run(prob, feed_dict={x: self.input_data['X_test']})

        #     predict = tf.argmax(prob, axis=1)
        #     # the prediction of the testing data
        #     predict_test = sess.run(predict, feed_dict={x:self.input_data['X_test']})

        #     # append the prediction of the testing data to self.predict_test
        #     self.predict_test.append(predict_test)

        #     utility = graph.get_tensor_by_name("logits:0")

        #     # market share
        #     self.mkt_share_train[index, :] = np.sum(prob_train, axis = 0) / self.numIndTrn
        #     self.mkt_share_test[index, :] = np.sum(prob_test, axis = 0) /self.numIndTest
        #     print(self.mkt_share_train[index, :])


        #     if disagg:
        #         # Disaggregate Prob Derivative and Elasticity
        #         grad = []
        #         drive_time = drive_time_idx
        #         drive_cost = drive_cost_idx
        #         for j in range(len(self.modes)):
        #             grad.append(tf.gradients(prob[:, j], x))

        #             # # training data (not necessary to calculate the vot on the training data)
        #             # gradients = sess.run(grad[-1], feed_dict={x: self.input_data['X_train']})[0]
        #             # # (variable, model #, mode, individual #) - elasticity/derivative
        #             # self.prob_derivative[:, index, j, :] = np.array(gradients[:, self.cont_vars]).T
        #             # elas = gradients[:, self.cont_vars] / prob_train[:, j][:, None] * np.array(self.X_train_standard)[:, self.cont_vars] / self.std[None,self.cont_vars]
        #             # self.elasticity[:, index, j, :] = np.array(elas).T

        #             # if j == drive_idx: # If drive, then calculate VOT
        #             #     # filter out bad records
        #             #     # print how many observations are filtered out
        #             #     # Value of time : dp/dtime over dp/dcost, correct for normalization as well as units
        #             #     # Take the mean of the trial
        #             #     v = gradients[:, drive_time] / gradients[:, drive_cost] / self.std[None, drive_time] * self.std[None, drive_cost] * time_correction
        #             #     self.vot_train[index,:] = v
        #             #     filt = ~np.isnan(v) & ~np.isinf(v)
        #             #     v = v[filt]
        #             #     print("Model ", index, ": dropped ", self.numIndTrn - len(v), " training observations.")
        #             #     self.filterd_train.append(self.numIndTrn - len(v))

        #             # testing data
        #             gradients = sess.run(grad[-1], feed_dict={x: self.input_data['X_test']})[0]
        #             self.prob_derivative_test[:, index, j, :] = np.array(gradients[:, self.cont_vars]).T
        #             elas = gradients[:, self.cont_vars] / prob_test[:, j][:, None] * np.array(self.X_test_standard)[:, self.cont_vars] / self.std[None,self.cont_vars]
        #             self.elasticity_test[:, index, j, :] = np.array(elas).T
                    
        #             if j == drive_idx: # If drive, then calculate VOT_drive
        #                 # filter out bad records
        #                 # print how many observations are filtered out
        #                 # Value of time : dp/dtime over dp/dcost, correct for normalization as well as units
        #                 # Take the mean of the trial
        #                 v = gradients[:, drive_time] / gradients[:, drive_cost] / self.std[None, drive_time] * self.std[None, drive_cost] * time_correction
        #                 self.vot_drive_test[index,:] = v
        #                 filt = ~np.isnan(v) & ~np.isinf(v)
        #                 v = v[filt]
        #                 print("Model ", index, ": dropped ", self.numIndTest - len(v), " testing observations.")
        #                 self.filterd_test.append(self.numIndTest - len(v))

        #                 util_derivative = tf.gradients(utility[:, j], x)
        #                 util_derivative_test = sess.run(util_derivative, feed_dict={x: self.input_data['X_test']})[0][:, self.cont_vars]
        #                 self.util_derivative_test[:, index, :] = np.array(util_derivative_test).T

        #             if j == pt_idx: # If pt, then calculate VOT_pt
        #                 # filter out bad records
        #                 # print how many observations are filtered out
        #                 # Value of time : dp/dtime over dp/dcost, correct for normalization as well as units
        #                 # Take the mean of the trial
        #                 v = gradients[:, pt_time_idx] / gradients[:, pt_cost_idx] / self.std[None, pt_time_idx] * self.std[None, pt_cost_idx] * time_correction
        #                 self.vot_pt_test[index,:] = v
        #                 filt = ~np.isnan(v) & ~np.isinf(v)
        #                 v = v[filt]
        #                 print("Model ", index, ": dropped ", self.numIndTest - len(v), " testing observations.")
        #                 self.filterd_test.append(self.numIndTest - len(v))

        #                 util_derivative = tf.gradients(utility[:, j], x)
        #                 util_derivative_test = sess.run(util_derivative, feed_dict={x: self.input_data['X_test']})[0][:, self.cont_vars]
        #                 self.util_derivative_test[:, index, :] = np.array(util_derivative_test).T

        #     if mkt:
        #         # Choice prob and Prob Derivative for market average person
        #         choice_prob, prob_derivative = self.compute_prob_curve(x, self.input_data['X_train'], self.cont_vars, prob, sess)
        #         # (variable, model #, mode, individual #) - choice prob
        #         self.mkt_prob_derivative[:, index, :, :] = np.array(prob_derivative)
        #         self.mkt_choice_prob[:, index, :, :] = np.array(choice_prob)

        #     if social_welfare:
        #         utility0 = sess.run(utility, feed_dict={x: self.input_data['X_test']})
        #         new_input = self.input_data['X_test'].copy()
        #         new_input[drive_cost_name] += 1
        #         utility1 = sess.run(utility, feed_dict={x: new_input})
        #         # sw_ind is welfare change per person
        #         sw_ind.append(np.log(np.sum(np.exp(utility1), axis=1)) - np.log(np.sum(np.exp(utility0), axis=1)))
        #     '''
        #     fig, ax = plt.subplots(figsize = (12, 12))
        #     ax.scatter(self.X_train_raw[:, 11], self.mkt_choice_prob[11, index, 3,:])
        #     fig.savefig('plots/test.png')
        #     '''

        #     sess.close()

        # self.average_cp = np.mean(self.mkt_choice_prob, axis = 1)
        # if social_welfare:
        #     j = drive_idx
        #     cost_var = drive_cost_idx_standard
        #     # welfare
        #     self.sw_ind = np.array(sw_ind)
        #     # definition of util_derivative_test
        #     # self.util_derivative_test = np.zeros((numContVars, self.numModels, self.numIndTest))
        #     self.sw_change_0 = sw_ind/self.util_derivative_test[cost_var,:,:] #, axis = 1)
        #     self.sw_change_1 = sw_ind/np.mean(self.util_derivative_test[cost_var,:,:], axis=0)[None, :]#, axis = 1)
        #     self.sw_change_2 = sw_ind/np.mean(self.util_derivative_test[cost_var,:,:], axis=1)[:, None]#, axis = 1)
        #     # sw_change_3 is used here? It is averaged over all iterations
        #     self.sw_change_3 = sw_ind/np.mean(self.util_derivative_test[cost_var,:,:])#, axis = 1)

    def preprocess(self, raw_data_dir, num_training_samples = None):
        """Preprocess the raw data

        Args:
            raw_data_dir (str): path to the raw data
            num_training_samples (int, optional): number of samples of training data. If not None, the first num_training_samples rows of training data would be used. Defaults to None.
        """        
        with open(raw_data_dir, 'rb') as f:
            data = pickle.load(f)
        if num_training_samples is None:
            X_train_raw = data['X_train'+self.suffix].values
            X_test_raw = data['X_test'+self.suffix].values
        else:
            X_train_raw = data['X_train'+self.suffix].values[:num_training_samples, :]
            X_test_raw = data['X_test'+self.suffix].values[:num_training_samples]

        # print(X_train_raw.shape)
        X_train_raw = pd.DataFrame(X_train_raw, columns = self.all_variables)
        X_test_raw = pd.DataFrame(X_test_raw, columns = self.all_variables)

        self.X_train_standard = X_train_raw
        self.X_test_standard = X_test_raw
        #X_train_nonstandard = X_train_raw[non_standard_vars]

        self.std = np.sqrt(StandardScaler().fit(pd.concat([self.X_train_standard, self.X_test_standard])).var_)
        #StandardScaler().fit(X_sp_train_standard).mean_[[8,9,10,14,15]]

        self.X_train_raw = np.array(X_train_raw)

    def write_result(self, result_dir, mean_result_file, model_result_file_prefix, model_result_file_suffix):
        """[summary]

        Args:
            result_dir (str): directory for saving result files
            mean_result_file (str): file name of the mean result
            model_result_file_prefix (str): prefix of the file name of separate model results
            model_result_file_suffix (str): suffix of the file name of separate model results
        """
        time_correction = 1

        b_write_each_model = True
        ## define file names (including path). The numbering of separate model starts from 0

        # define the file names for each model result
        mean_result_file = os.path.join(result_dir, mean_result_file)
        if b_write_each_model is True:
            list_sep_model_result_file = list(map(lambda x: os.path.join(result_dir, model_result_file_prefix + str(x) + model_result_file_suffix), range(self.numModels)))

        print(list_sep_model_result_file[0])
        print(os.path.exists(list_sep_model_result_file[0]))
        #######
        # WRITE RESULTS TO FILES
        # 1. open (N + 1) files, with N representing the model numbers 
        ## clean the files
        print("Remove files before writing results")
        if os.path.exists(mean_result_file):
            os.remove(mean_result_file)
            if b_write_each_model is True:
                    for model_res_file in list_sep_model_result_file:
                            if os.path.exists(model_res_file):
                                    os.remove(model_res_file)

        ## open the files in appending mode
        hdl_mean_result_file = open(mean_result_file, 'a')
        if b_write_each_model is True:
            list_dhl_sep_model_result_file = list(map(lambda x: open(x, 'a'), list_sep_model_result_file))

        # 2. write to a file

        # define a function to write to a file
        # transform the np array to a df, set index and column names, and then append to a csv
        def write_info_block(hdl_file, str_info, nparray_2d_content, list_index_name, list_col_names):
            hdl_file.write(str_info)
            tmp_df = pd.DataFrame(data=nparray_2d_content, index = list_index_name, columns = list_col_names)
            tmp_df.to_csv(hdl_file, mode='a', index=True)
            hdl_file.write("\n")

        ## For each metric, write a sentence, and then append the numpy array to files 
        ### 2.1 write confusion matrix and test_accuracy
        mean_overall_metrics, mean_class_metrics, mean_confusion_matrix, list_overall_metrics, list_class_metrics, list_confusion_matrix = self.calculate_accuracy_metrics_confusion_matrix(self.input_data['Y_test'], self.predict_test)
        
        intro = "# overall_metrics\n"
        list_index_name = ["accuracy", 'macro_f1_score', 'macro_recall']
        list_col_name = ["overall"]
        content = mean_overall_metrics
        content_all_model = list_overall_metrics

        write_info_block(hdl_mean_result_file, intro, content, list_index_name, list_col_name)
        if b_write_each_model is True:
            # average over individuals
            for ind, hdl in zip(range(self.numModels), list_dhl_sep_model_result_file):
                    content = content_all_model[ind]
                    write_info_block(hdl, intro, content, list_index_name, list_col_name)
        
        intro = "# class_specific_metrics\n"
        list_index_name = mean_class_metrics.index
        list_col_name = mean_class_metrics.columns
        content = mean_class_metrics
        content_all_model = list_class_metrics

        write_info_block(hdl_mean_result_file, intro, content, list_index_name, list_col_name)
        if b_write_each_model is True:
            # average over individuals
            for ind, hdl in zip(range(self.numModels), list_dhl_sep_model_result_file):
                    content = content_all_model[ind]
                    write_info_block(hdl, intro, content, list_index_name, list_col_name)
        
        intro = "# confusion_matrix\n"
        list_index_name = self.modes
        list_col_name = self.modes
        content = mean_confusion_matrix
        content_all_model = list_confusion_matrix  

        write_info_block(hdl_mean_result_file, intro, content, list_index_name, list_col_name)
        if b_write_each_model is True:
            # average over individuals
            for ind, hdl in zip(range(self.numModels), list_dhl_sep_model_result_file):
                    content = content_all_model[ind]
                    write_info_block(hdl, intro, content, list_index_name, list_col_name)

        ### 2.2 write prob_derivative
        intro = "# prob_derivative\n"
        list_index_name = self.cont_vars_name
        list_col_name = self.modes
        # self.prob_derivative_test = np.zeros((numContVars, self.numModels, self.numAlt, self.numIndTest))
        content = np.mean(self.prob_derivative_test, axis=(1,3))

        print(content.shape) # should be (11, 4)
        write_info_block(hdl_mean_result_file, intro, content, list_index_name, list_col_name)
        if b_write_each_model is True:
            # average over individuals
            content_all_model = np.mean(self.prob_derivative_test, axis=3)
            for ind, hdl in zip(range(self.numModels), list_dhl_sep_model_result_file):
                    content = content_all_model[:,ind,:]
                    write_info_block(hdl, intro, content, list_index_name, list_col_name)

        ### 2.3 write elasticity
        intro = "# elasticity\n"
        list_index_name = self.cont_vars_name
        list_col_name = self.modes
        # self.elasticity = np.zeros((numContVars, self.numModels, self.numAlt, self.numIndTest))
        content = np.mean(self.elasticity_test, axis=(1,3))
        content_all_model = np.mean(self.elasticity_test, axis=3)

        print(content.shape) # should be (11, 4)
        write_info_block(hdl_mean_result_file, intro, content, list_index_name, list_col_name)
        if b_write_each_model is True:
            # average over individuals
            
            for ind, hdl in zip(range(self.numModels), list_dhl_sep_model_result_file):
                    content = content_all_model[:,ind,:]
                    write_info_block(hdl, intro, content, list_index_name, list_col_name)

        ### 2.4 write market share
        intro = "# market share\n"
        list_index_name = ["market_share"]
        list_col_name = self.modes
        # self.mkt_share_test = np.zeros((self.numModels, self.numAlt))
        content = np.mean(self.mkt_share_test, axis = 0, keepdims = True)
        content_all_model = self.mkt_share_test

        print(content.shape)
        write_info_block(hdl_mean_result_file, intro, content, list_index_name, list_col_name)
        if b_write_each_model is True:
            # average over individuals
            for ind, hdl in zip(range(self.numModels), list_dhl_sep_model_result_file):
                    # using None to avoid losing dimensions. Would get a np array of [1, self.numAlt]
                    content = content_all_model[None,ind,:]
                    write_info_block(hdl, intro, content, list_index_name, list_col_name)

        ### 2.4 write VOT
        intro = "# value of time of driving\n"
        list_index_name = ["VOT_drive"]
        list_col_name = ["VOT_drive"]
        # self.vot_test = np.zeros((self.numModels, self.numIndTest))
        content = np.ma.masked_invalid(self.vot_drive_test).mean()
        # content_all_model is a 0d array
        content_all_model = np.ma.masked_invalid(self.vot_drive_test).mean(axis = 1)

        # print(content.shape)
        write_info_block(hdl_mean_result_file, intro, content, list_index_name, list_col_name)
        if b_write_each_model is True:
            # average over individuals
            for ind, hdl in zip(range(self.numModels), list_dhl_sep_model_result_file):
                    content = content_all_model[ind]
                    write_info_block(hdl, intro, content, list_index_name, list_col_name)

        intro = "# value of time of pt\n"
        list_index_name = ["VOT_pt"]
        list_col_name = ["VOT_pt"]
        # self.vot_test = np.zeros((self.numModels, self.numIndTest))
        content = np.ma.masked_invalid(self.vot_pt_test).mean()
        # content_all_model is a 0d array
        content_all_model = np.ma.masked_invalid(self.vot_pt_test).mean(axis = 1)

        # print(content.shape)
        write_info_block(hdl_mean_result_file, intro, content, list_index_name, list_col_name)
        if b_write_each_model is True:
            # average over individuals
            for ind, hdl in zip(range(self.numModels), list_dhl_sep_model_result_file):
                    content = content_all_model[ind]
                    write_info_block(hdl, intro, content, list_index_name, list_col_name)

        # 3. close the files
        hdl_mean_result_file.close()
        if b_write_each_model is True:
            for x in list_dhl_sep_model_result_file:
                x.close()

    def pickle_model(self, model_pickle_file):
        """Save the model as a pickle file. Used as input to plots

        Args:
            model_pickle_file (str): path to the pickle file
        """        
        with open(model_pickle_file, 'wb') as f:
            pickle.dump(self, f)            
    
    def calculate_accuracy_metrics_confusion_matrix(self, Y_true, list_Y_predict):
        """Calculate overall metrics (accuracy, macro average F1 score, macro average recall), class-specific metrics (precision, recall, F1 score), and confusion matrix for several prediction results

        Args:
            Y_true (list): True labels
            list_Y_predict (list of list of int): A list of prediction results (a list of int)
        Returns:
            [mean_overall_metrics, mean_class_metrics, mean_confusion_matrix, list_overall_metrics, list_class_metrics, list_confusion_matrix]
        """
        list_overall_metrics = []
        list_class_metrics = []
        for y_pred in list_Y_predict:
            test_report = classification_report(Y_true, y_pred, target_names=self.modes, output_dict=True)
            # a list of accuracy, macro_f1_score, macro_recall
            list_overall_metrics.append([test_report['accuracy'], test_report['macro avg']['f1-score'], test_report['macro avg']['recall']])
            list_class_metrics.append(pd.DataFrame.from_dict({k: test_report[k] for k in self.modes}))

        mean_overall_metrics = np.mean(list_overall_metrics, axis = 0)
        mean_class_metrics = np.mean(list_class_metrics, axis = 0)
        mean_class_metrics = pd.DataFrame(mean_class_metrics, index = list_class_metrics[0].index, columns = list_class_metrics[0].columns)

        list_confusion_matrix = list(map(lambda x:confusion_matrix(Y_true, x), list_Y_predict))
        mean_confusion_matrix = np.mean(list_confusion_matrix, axis = 0)
        return [mean_overall_metrics, mean_class_metrics, mean_confusion_matrix, list_overall_metrics, list_class_metrics, list_confusion_matrix]


