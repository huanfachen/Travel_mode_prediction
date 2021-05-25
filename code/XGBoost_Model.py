#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thr May 06 17:16:41 2021

@author: Huanfa Chen
"""

import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import xgboost 
import copy
import os
import pickle as pkl
import random
import datetime

class XGBoost_Model:
    def __init__(self,num_alt,MODEL_NAME,INCLUDE_VAL_SET,INCLUDE_RAW_SET, RUN_DIR):
        self.num_alt = num_alt
        self.MODEL_NAME = MODEL_NAME
        self.INCLUDE_VAL_SET = INCLUDE_VAL_SET
        self.INCLUDE_RAW_SET=INCLUDE_RAW_SET
        self.RUN_DIR = RUN_DIR
        self.FILE_XGBOOST_MODEL = os.path.join(self.RUN_DIR, "{}.pkl".format(MODEL_NAME))
#    MODEL_NAME = 'model'
#    INCLUDE_VAL_SET = False
#    input_file="data/SGP_SP.pickle"
#    INCLUDE_RAW_SET = True
#    self.K = 2
    
    def init_hyperparameter(self, rand):
        """Initialise the hyperparameters

        Args:
            rand (bool): if True, the hyperparameters will be initialised randomly. Otherwise, the tuned optimal hyperparameters will be used.
        """        
        # space={'max_depth': hp.quniform("max_depth", 1, 11, 1),
        # 'gamma': hp.uniform ('gamma', 0,5),
        # 'min_child_weight' : hp.quniform('min_child_weight', 1, 11, 1),
        # 'eta': hp.uniform('eta', 0.0, 1.0),
        # 'n_estimators': hp.quniform('n_estimators', 100, 500, 50),
        # }
        # h stands for hyperparameter
        self.h = {}
        # default value
        self.h['subsample'] = 0.7
        self.h['colsample_bytree'] = 0.7
        self.h['colsample_bylevel'] = 0.7
        # random state
        self.h['random_state'] = random.seed(datetime.datetime.now())

        if rand:
            self.h['max_depth']=np.random.choice(self.hs['max_depth'])
            self.h['gamma']=np.random.choice(self.hs['gamma'])
            self.h['min_child_weight']=np.random.choice(self.hs['min_child_weight'])
            self.h['eta']=np.random.choice(self.hs['eta'])
            self.h['n_estimators']=np.random.choice(self.hs['n_estimators'])
        else:
            # using the optimal hyperparameter
            self.h['max_depth']=10
            self.h['gamma']=0.5580
            self.h['min_child_weight']=1
            self.h['eta']=0.0087
            self.h['n_estimators']=300
        # self.h['batch_normalization']=True
        # self.h['learning_rate']=1e-3
        # self.h['n_iteration']=5000
        # self.h['n_mini_batch']=200

    def change_hyperparameter(self, new_hyperparameter):
        assert bool(self.h) == True
        self.h = new_hyperparameter
    
    def init_hyperparameter_space(self):
        # hs stands for hyperparameter_space
        self.hs = {}
        
        # maximum depth of the individual regression estimators        
        self.hs['max_depth']=[1,2,3,4,5,6,7,8,9,10,11] 
        # aka min_split_loss, minimum loss reduction required to make a further partition on a leaf node of the tree
        self.hs['gamma']=[0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        # Minimum child weight. The larger min_child_weight, the less likely to overfit
        self.hs['min_child_weight']=[1,2,3,4,5,6,7,8,9,10,11]
        # 
        self.hs['eta']=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # number of trees
        self.hs['n_estimators']=[100, 150, 200, 250, 300]

    def random_sample_hyperparameter(self):
        assert bool(self.hs) == True
        assert bool(self.h) == True
        for name_ in self.h.keys():
            self.h[name_] = np.random.choice(self.hs[name_+'_list'])

    # def obtain_mini_batch(self):
    #     index = np.random.choice(self.N_bootstrap_sample, size = self.h['n_mini_batch'])   
    #     self.X_batch = self.X_train_[index, :]
    #     self.Y_batch = self.Y_train_[index]
        
    def load_data(self, input_data, input_data_raw = None, num_training_samples = None):
        """Load data

        Args:
            input_data (list): containing numpy arrays of training, validation, and testing data
            input_data_raw (list, optional): containing numpy arrays of raw training, validation, and testing data. Defaults to None.
            num_training_samples (int, optional): the number of records for model training. If None, all training data in input_data are used. Defaults to None.
        """        
        # print("Loading datasets...")
        self.colnames = list(input_data['X_train'].columns)
        if num_training_samples is None:
            self.X_train = input_data['X_train'].values
            self.Y_train = input_data['Y_train'].values
        else:
            self.X_train = input_data['X_train'].values[:num_training_samples, :]
            self.Y_train = input_data['Y_train'].values[:num_training_samples]
        self.X_test=input_data['X_test'].values
        self.Y_test=input_data['Y_test'].values
        if self.INCLUDE_VAL_SET:
            self.X_val = input_data['X_val'].values
            self.Y_val = input_data['Y_val'].values

        if self.INCLUDE_RAW_SET:
            if num_training_samples is None:
                self.X_train_raw = input_data_raw['X_train'].values
                self.Y_train_raw = input_data_raw['Y_train'].values
            else:
                self.X_train_raw = input_data_raw['X_train'].values[:num_training_samples, :]
                self.Y_train_raw = input_data_raw['Y_train'].values[:num_training_samples]

            self.X_test_raw=input_data_raw['X_test'].values
            self.Y_test_raw=input_data_raw['Y_test'].values                
            if self.INCLUDE_VAL_SET:
                self.X_val_raw = input_data_raw['X_val'].values
                self.Y_val_raw = input_data_raw['Y_val'].values
                
        # print("Training set", self.X_train.shape, self.Y_train.shape)
        # print("Testing set", self.X_test.shape, self.Y_test.shape)
        # if self.INCLUDE_VAL_SET:
        #     print("Validation set", self.X_val.shape, self.Y_val.shape)
        # save dim
        self.N_train,self.D = self.X_train.shape
        self.N_test,self.D = self.X_test.shape

    def bootstrap_data(self, N_bootstrap_sample):
        """
        This function samples from training set with replacement. Boostrap will not change the data distribution.
        """
        print("Bootstrap ", N_bootstrap_sample, " samples from training set...")
        self.N_bootstrap_sample = N_bootstrap_sample
        bootstrap_sample_index = np.random.choice(self.N_train, size = self.N_bootstrap_sample) 
        self.X_train_ = self.X_train[bootstrap_sample_index, :]
        self.Y_train_ = self.Y_train[bootstrap_sample_index]

    # def standard_hidden_layer(self, name):
    #     # standard layer, repeated in the following for loop.
    #     self.hidden = tf.layers.dense(self.hidden, self.h['n_hidden'], activation = tf.nn.relu, name = name)
    #     if self.h['batch_normalization'] == True:
    #         self.hidden = tf.layers.batch_normalization(inputs = self.hidden, axis = 1)
    #     self.hidden = tf.layers.dropout(inputs = self.hidden, rate = self.h['dropout_rate'])

    def build_model(self):
        self.xgb_clf = xgboost.XGBClassifier(max_depth = self.h['max_depth'],
        gamma=self.h['gamma'],
        min_child_weight=self.h['min_child_weight'],
        eta=self.h['eta'],
        n_estimators=self.h['n_estimators'],
        subsample=self.h['subsample'],
        colsample_bytree=self.h['colsample_bytree'],
        colsample_bylevel=self.h['colsample_bylevel'],
        random_state=self.h['random_state']
        )
                        
    def train_model(self):
        """Train and save the model
        """
        self.xgb_clf.fit(self.X_train, self.Y_train)
        # dump
        with open(self.FILE_XGBOOST_MODEL, 'wb') as f:
              pkl.dump(self.xgb_clf, f)

        # with tf.Session(graph=self.graph) as sess:
        #     self.init.run()
        #     for i in range(self.h['n_iteration']):
        #         if i%500==0:
        #             print("Iteration ", i, "Cost = ", self.cost.eval(feed_dict = {self.X: self.X_train_, self.Y: self.Y_train_}))
        #         # gradient descent
        #         self.obtain_mini_batch()
        #         sess.run(self.training_op, feed_dict = {self.X: self.X_batch, self.Y: self.Y_batch})
            
        #     ## compute accuracy and loss
        #     self.accuracy_train = self.accuracy.eval(feed_dict = {self.X: self.X_train_, self.Y: self.Y_train_})
        #     self.accuracy_test = sess.run(self.accuracy, feed_dict = {self.X: self.X_test, self.Y: self.Y_test})
        #     self.loss_train = self.cost.eval(feed_dict = {self.X: self.X_train_, self.Y: self.Y_train_})
        #     self.loss_test = self.cost.eval(feed_dict = {self.X: self.X_test, self.Y: self.Y_test})
        #     if self.INCLUDE_VAL_SET:
        #         self.accuracy_val = self.accuracy.eval(feed_dict = {self.X: self.X_val, self.Y: self.Y_val})
        #         self.loss_val = self.cost.eval(feed_dict = {self.X: self.X_val, self.Y: self.Y_val})

        #     ## compute util and prob
        #     self.util_train=self.output.eval(feed_dict={self.X: self.X_train, self.Y: self.Y_train})
        #     self.util_test=self.output.eval(feed_dict={self.X: self.X_test, self.Y: self.Y_test})
        #     self.prob_train=self.prob.eval(feed_dict={self.X: self.X_train, self.Y: self.Y_train})
        #     self.prob_test=self.prob.eval(feed_dict={self.X: self.X_test, self.Y: self.Y_test})
        #     if self.INCLUDE_VAL_SET:
        #         self.util_val=self.output.eval(feed_dict={self.X: self.X_val, self.Y: self.Y_val})
        #         self.prob_val=self.prob.eval(feed_dict={self.X: self.X_val, self.Y: self.Y_val})
        #     ## save
        #     self.saver.save(sess, os.path.join(self.RUN_DIR, self.MODEL_NAME+".ckpt"))

    def init_simul_data(self):
        self.simul_data_dic = {}

    def create_one_simul_data(self, x_col_name, x_delta):
        # create a dataset in which only targetting x is ranging from min to max. All others are at mean value.
        # add it to the self.simul_data_dic
        # use min and max values in testing set to create the value range
        target_x_index = self.colnames.index(x_col_name)
        self.N_steps = np.int((np.max(self.X_train[:,target_x_index]) - np.min(self.X_train[:,target_x_index]))/x_delta) + 1
        data_x_target_varying = np.tile(np.mean(self.X_train, axis = 0), (self.N_steps, 1))
        data_x_target_varying[:, target_x_index] = np.arange(np.min(self.X_train[:,target_x_index]), np.max(self.X_train[:,target_x_index]), x_delta)
        self.simul_data_dic[x_col_name] = data_x_target_varying
        
    def compute_simul_data(self):
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, "tmp/"+self.MODEL_NAME+".ckpt")
            # compute util and prob
            self.util_simul_dic={}
            self.prob_simul_dic={}
            for name_ in self.simul_data_dic.keys():
                self.util_simul_dic[name_]=self.output.eval(feed_dict={self.X:self.simul_data_dic[name_]})
                self.prob_simul_dic[name_]=self.prob.eval(feed_dict={self.X:self.simul_data_dic[name_]})

    def init_x_delta_data(self):
        self.x_delta_data_dic = {}
        
    def create_one_x_delta_data(self, x_col_name, x_delta):
        # create a dataset in which only targetting x_col becomes x + delta. All the others are the SAME as x. 
        # by default, we focus on training set.
        # add the new dataset to the self.x_delta_data_dic
        target_x_index = self.colnames.index(x_col_name)
        x_delta_data = copy.copy(self.X_train)
        x_delta_data[:, target_x_index] = x_delta_data[:, target_x_index] + x_delta # add delta to the X_train dataset in x_col_name column
        self.x_delta_data_dic[x_col_name]=x_delta_data
        
    # def compute_x_delta_data(self):
    #     with tf.Session(graph=self.graph) as sess:
    #         self.saver.restore(sess, "tmp/"+self.MODEL_NAME+".ckpt")
    #         # compute util and prob
    #         self.util_x_delta_dic={}
    #         self.prob_x_delta_dic={}
    #         for name_ in self.x_delta_data_dic.keys():
    #             self.util_x_delta_dic[name_]=self.output.eval(feed_dict={self.X:self.x_delta_data_dic[name_]})
    #             self.prob_x_delta_dic[name_]=self.prob.eval(feed_dict={self.X:self.x_delta_data_dic[name_]})
                            
    def visualize_choice_prob_function(self, x_col_name, color_list, label_list, xlabel, ylabel):
        assert len(color_list)==self.num_alt
        assert len(label_list)==self.num_alt
        # targeting x index.
        target_x_index = self.colnames.index(x_col_name)







