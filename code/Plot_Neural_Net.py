# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 13:00:00 2021

@author: Huanfa Chen

Based on https://github.com/cjsyzwsh/dnn-for-economic-information/blob/master/code/3_plot.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os

class Plot_Neural_Net:
    """
    Reading in the neural network model (in pickle format) and calculating relevant economic metrics
    """
    def __init__(self, model_pkl_file, dir_plots, data_name, method_name):
        """Initialise this class object

        Args:
            model_pkl_file (str): path to the model pickle file
            dir_plots (str): directory path for saving plots
            data_name (str): dataset name 
            method_name (str): method name
        """        

        with open(model_pkl_file, 'rb') as f:
            self.m = pickle.load(f)
        # name suffix (aka the format)
        self.name_plot_suffix = '.png'
        self.dir_plots = dir_plots
        self.data_name = data_name
        self.method_name = method_name
        # define name of the plots
        # self.name_plot_prefix = os.path.join(dir_plots, '_'.join([data_name, method_name, 'mean']))
        # # static plot names
        # self.name_accuracy_hist = os.path.join(dir_plots, "{data}_{method}_{model}_{plot_type}_{suffix}".format(data_name, method_name, 'mean', 'accuracy_hist',self.name_plot_suffix)
        # self.name_vot_hist = os.path.join(dir_plots, "{data}_{method}_{model}_{plot_type}_{suffix}".format(data_name, method_name, 'mean', 'vot_hist',self.name_plot_suffix)  
        # # name of choice prob plot. One mode and one independent variable. 
        # self.name_choice_prob_line_prefix = os.path.join(dir_plots, '_'.join([data_name, method_name, 'mean', 'choice_prob_line']))
        # # name of choice prob derivative plot. One mode and one independent variable.
        # self.name_choice_prob_derivative_line_prefix = os.path.join(dir_plots, '_'.join([data_name, method_name, 'mean', 'choice_prob_derivative_line']))
        # # name of substitute pattern plot. All modes and one independent variable
        # self.name_substitute_pattern_line_prefix = os.path.join(dir_plots, '_'.join([data_name, method_name, 'mean', 'choice_prob_line']))

    def save_plot(self, fig_obj, plot_type, mode="", var=""):
        """Save figures with formatted names 

        Args:
            fig_obj ([type]): [description]
            plot_type ([type]): [description]
            mode ([type], optional): [description]. Defaults to None.
            var ([type], optional): [description]. Defaults to None.
        """        
        # fig_name = None
        fig_name = "{data}_{method}_{model}_{plot_type}_{mode}_{var}_{suffix}".format(data = self.data_name, method = self.method_name, model = 'mean', \
            plot_type = plot_type, mode = mode, var = var, suffix = self.name_plot_suffix)
        fig_obj.savefig(os.path.join(self.dir_plots, fig_name), bbox_inches="tight")    

    def plot_choice_prob(self, plot_modes, plot_vars, plot_vars_standard, plot_var_names, highlight=[], highlightlabel=[]):
        """Plot the choice probabilities of driving with varying driving costs. See Fig.4 in the paper.
        The mkt_choice_prob is used here.
        Note that plot_modes and plot_vars should be paired and have the same length. 

        Args:
            m (object): the analysis object 
            plot_modes (list of str): containing modes, e.g. 'walk', 'pt', 'cycle', 'drive'
            plot_vars (list of int): containing the index of plot_vars in all_vars 
            plot_vars_standard (list of int): containing the index of plot_vars in standard_vars
            plot_var_names (list of str): containing labels on the x axis, e.g. 'drive_cost ($)'
            highlight (list, optional): [description]. Defaults to [].
            highlightlabel (list, optional): [description]. Defaults to [].
        """    
        colors = ['red','darkorange','darkgreen','darkorchid','blue']
        axes_cp = []
        for i in range(len(plot_vars)):
            axes_cp.append(plt.subplots(figsize=(8, 8)))
            mode = plot_modes[i]
            axes_cp[i][1].set_ylabel(mode + " probability")
            axes_cp[i][1].set_xlabel(plot_var_names[i])

        for i, v, vs in zip([x for x in range(len(plot_vars))], plot_vars, plot_vars_standard):
            # mode for plotting
            mode = plot_modes[i]
            j = self.m.modes.index(mode)
            for index in range(self.m.numModels):
                plot = sorted(zip(self.m.X_train_raw[:, v], self.m.mkt_choice_prob[vs, index, j, :]))
                if index not in highlight:
                    axes_cp[i][1].plot([x[0] for x in plot], [x[1] for x in plot], linewidth=1, color='silver',label='')
            for index in highlight:
                plot = sorted(zip(self.m.X_train_raw[:, v], self.m.mkt_choice_prob[vs, index, j, :]))
                axes_cp[i][1].plot([x[0] for x in plot], [x[1] for x in plot], linewidth=1, color=colors[highlight.index(index)], label = highlightlabel[highlight.index(index)])
            # plot = sorted(zip(self.m.X_train_raw[:, v], self.m.average_cp[vs, j, :]))
            # axes_cp[i][1].plot([x[0] for x in plot], [x[1] for x in plot], linewidth=3, color='k',label='Average')

            df = pd.DataFrame(np.array([self.m.X_train_raw[:, v], self.m.average_cp[vs, j, :]]).T, columns=['var', 'cp']).groupby('var', as_index=False).mean()[['var', 'cp']]
            df.sort_values(by='var', inplace=True)
            axes_cp[i][1].plot(df['var'], df['cp'], linewidth=3, color='k',label='Average')

            axes_cp[i][1].set_xlim([0, np.percentile(self.m.X_train_raw[:, v], 99)])
            axes_cp[i][1].set_ylim([0, 1])
            axes_cp[i][1].legend(fancybox=True,framealpha = 0.5)
            var_name = self.m.all_variables[v]
            self.save_plot(axes_cp[i][0], 'choice_prob', mode, var_name)

    def plot_substitute_pattern(self, plot_vars, plot_vars_standard, plot_var_names):
        """Plot the choice probabilities of all alternatives with varying driving costs. See Fig.4 in the paper.
        The mkt_choice_prob is used here.

        Args:
            m (object): the analysis object 
            plot_vars (list of int): containing the index of plot_vars in all_vars 
            plot_vars_standard (list of int): containing the index of plot_vars in standard_vars
            plot_var_names (list of str): containing labels on the x axis, e.g. 'drive_cost ($)'
        """    
        colors = ['red','darkorange','darkgreen','darkorchid','blue']
        for name, v, vs in zip(plot_var_names, plot_vars, plot_vars_standard):
            fig, ax = plt.subplots(figsize=(8, 8))
            for j in range(self.m.numAlt):
                df = np.insert(self.m.mkt_choice_prob[vs, :, j, :], 0, self.m.X_train_raw[:, v], axis=0)
                df = df[:, df[0, :].argsort()]
                for index in range(self.m.numModels):
                    ax.plot(df[0, :], df[index + 1, :], linewidth=1, alpha=0.5, color=self.m.colors[j], label='')
                
            for j in range(self.m.numAlt):
                # the mkt_choice_prob is used here to plot the choice probability choice with varying driving cost.
                # in mkt_choice_prob, all variables uses the market average value, except for the targeted variable
                plot = sorted(zip(self.m.X_train_raw[:, v], np.mean(self.m.mkt_choice_prob[vs, :, j, :], axis = 0)))
                ax.plot([x[0] for x in plot], [x[1] for x in plot], linewidth=3, color=colors[j], label=self.m.modes[j])
            #ax.legend()
            ax.set_ylabel("choice probability")
            ax.set_xlabel(name)
            ax.set_xlim([0, np.percentile(self.m.X_train_raw[:, v], 99)])
            ax.set_ylim([0, 1])
            # getting the variable name from m.all_variables
            var_name = self.m.all_variables[v]
            self.save_plot(fig, 'subtitute_pattern', var_name)

    def plot_prob_derivative(self, plot_modes, plot_vars, plot_vars_standard, plot_var_names, highlight = [], highlightlabel = []):
        """Plot probability derivative. mkt_prob_derivative is used here.
        Note that plot_modes and plot_vars should be paired and have the same length. 

        Args:
            m (object): the analysis object
            plot_modes (list of str): containing modes, e.g. 'walk', 'pt', 'cycle', 'drive' 
            plot_vars (list of int): containing the index of plot_vars in all_vars 
            plot_vars_standard (list of int): containing the index of plot_vars in standard_vars
            plot_var_names (list of str): containing labels on the x axis, e.g. 'drive_cost ($)'
            highlight (list of int, optional): list containing the index of highlighted models. Defaults to [].
            highlightlabel (list of str, optional): list containing the index of highlighted models. Defaults to [].
        """    
        colors = ['red','darkorange','darkgreen','darkorchid','blue']
        axes_pd = []
        for i in range(len(plot_vars)):
            axes_pd.append(plt.subplots(figsize=(8, 8)))
            mode = plot_modes[i]
            axes_pd[i][1].set_ylabel(mode + " probability derivative")
            axes_pd[i][1].set_xlabel(plot_var_names[i])

        for i, v, vs in zip([x for x in range(len(plot_vars))], plot_vars, plot_vars_standard):
            mode = plot_modes[i]
            j = self.m.modes.index(mode)
            for index in range(self.m.numModels):
                # (variable, model  # , mode, individual #)
                df = pd.DataFrame(np.array([self.m.X_train_raw[:, v], self.m.mkt_prob_derivative[vs, index, j, :]]).T,
                                columns=['var', 'pd']).groupby('var', as_index=False).mean()[['var', 'pd']]
                df.sort_values(by='var', inplace=True)
                if index not in highlight:
                    axes_pd[i][1].plot(df['var'], df['pd'], linewidth=2, color='silver', label='')
            for index in highlight:
                df = pd.DataFrame(np.array([self.m.X_train_raw[:, v], self.m.mkt_prob_derivative[vs, index, j, :]]).T,
                                columns=['var', 'pd']).groupby('var', as_index=False).mean()[['var', 'pd']]
                df.sort_values(by='var', inplace=True)
                axes_pd[i][1].plot(df['var'], df['pd'], linewidth=2, color=colors[highlight.index(index)], label = highlightlabel[highlight.index(index)])

            # interval = (np.percentile(self.m.X_train_raw[:, v], 95) - self.m.X_train_raw[:, v].min()) / 50
            # temp = self.m.X_train_raw[:, v] // interval * interval
            average_pd = np.mean(self.m.mkt_prob_derivative, axis=1)
            df = pd.DataFrame(np.array([self.m.X_train_raw[:, v], average_pd[vs, j, :]]).T, columns=['var', 'pd']).groupby('var', as_index=False).mean()[['var', 'pd']]
            df.sort_values(by='var', inplace=True)
            axes_pd[i][1].plot(df['var'], df['pd'], linewidth=3, color='k',label='Average')
            axes_pd[i][1].set_xlim([0, np.percentile(self.m.X_train_raw[:, v], 99)])
            #axes_pd[i][1].set_ylim([-15, 5])
            axes_pd[i][1].legend(fancybox=True,framealpha = 0.5)
            # getting the variable name from m.all_variables
            var_name = self.m.all_variables[v]
            self.save_plot(axes_pd[i][0], 'choice_prob_derivative', mode, var_name)
        '''
        pos_1 = []
        pos_1_cnt = []
        neg_1 = []
        neg_1_cnt = []
        for j in range(5):
            neg_1.append(np.argmax(np.sum(rank_el[:, j, :] == 0, axis=1)))
            pos_1.append(np.argmax(np.sum(rank_el[:, j, :] == len(self.m.cont_vars) - 1, axis=1)))
            neg_1_cnt.append(np.max(np.sum(rank_el[:, j, :] == 0, axis=1)))
            pos_1_cnt.append(np.max(np.sum(rank_el[:, j, :] == len(self.m.cont_vars) - 1, axis=1)))
            print('Mode: ', self.m.mode[j], '\n\t', self.m.all_variables[pos_1[-1]], '(', pos_1_cnt[-1] / self.m.numModels, ')', \
                self.m.all_variables[neg_1[-1]], '(', neg_1_cnt[-1] / self.m.numModels, ')')
        '''

    def plot_onemodel_prob_derivative(self, modelNumber, plot_var, plot_vars_standard, plot_var_names):
        """Not tested or used.

        Args:
            modelNumber ([type]): [description]
            plot_var ([type]): [description]
            plot_vars_standard ([type]): [description]
            plot_var_names ([type]): [description]
        """        
        for name, v, vs in zip(plot_var_names, plot_var, plot_vars_standard):
            fig, ax = plt.subplots(figsize=(8, 8))
            mode = name[:name.find('_')]
            j = self.m.modes.index(mode)
            temp = np.concatenate(
                (self.m.prob_derivative[vs, modelNumber, j, :], self.m.prob_derivative_test[vs, modelNumber, j, :]))
            bins = np.linspace(np.percentile(temp, 1), np.percentile(temp, 99), 20)
            ax.hist(self.m.prob_derivative[vs, modelNumber, j, :], bins=bins, density=True, color='b')
            ax.set_xlabel(name + " probability derivative")
            ax.set_ylabel('Density')
            ax.set_xlim([np.percentile(temp, 1), np.percentile(temp, 99)])
            fig.savefig('plots/' + self.m.run_dir + '_' + str(modelNumber) + '_prob_derivative_train_' + str(v) + '.png', bbox_inches="tight")

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.hist(self.m.prob_derivative_test[vs, modelNumber, j, :], bins=bins, density=True, color='b')
            ax.set_xlabel(name + " probability derivative")
            ax.set_ylabel('Density')
            ax.set_xlim([np.percentile(temp, 1), np.percentile(temp, 99)])
            fig.savefig('plots/' + self.m.run_dir + '_' + str(modelNumber) + '_prob_derivative_test_' + str(v) + '.png', bbox_inches="tight")


    def plot_onemodel_vot(self, modelNumber, currency):
        """Not tested or used.

        Args:
            modelNumber ([type]): [description]
            currency ([type]): [description]
        """        
        temp = np.concatenate((self.m.vot_test[modelNumber, :], self.m.vot_train[modelNumber, :]))
        bins = np.linspace(np.percentile(temp, 5), np.percentile(temp, 95),
                        int((np.percentile(temp, 95) - np.percentile(temp, 5)) / 3))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.hist(self.m.vot_test[modelNumber, :], bins=bins, density=True, color='b')
        #ax.set_xlim([np.percentile(temp, 5), np.percentile(temp, 95)])
        #ax.set_xlim([-200, 200])
        ax.set_xlabel("Value of Time (Drive) ("+currency+"/h)")
        ax.set_ylabel('Density')
        fig.savefig('plots/' + self.m.run_dir + '_' + str(modelNumber) + '_vot_drive_test.png', bbox_inches="tight")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.hist(self.m.vot_train[modelNumber, :], bins=bins, density=True, color='b')
        print('(Train) Max Value:', np.max(self.m.vot_train[modelNumber, :]))
        print('(Train) Min Value:', np.min(self.m.vot_train[modelNumber, :]))
        print('(Train) 5th percentile Value:', np.percentile(self.m.vot_train[modelNumber, :], 5))
        print('(Train) 95th percentile Value:', np.percentile(self.m.vot_train[modelNumber, :], 95))
        print("Model ", modelNumber, " VOT mean: ", np.mean(self.m.vot_train[modelNumber, :]))
        print("Model ", modelNumber, " VOT median: ", np.median(self.m.vot_train[modelNumber, :]))
        print('(Test) Max Value:', np.max(self.m.vot_test[modelNumber, :]))
        print('(Test) Min Value:', np.min(self.m.vot_test[modelNumber, :]))
        print('(Test) 5th percentile Value:', np.percentile(self.m.vot_test[modelNumber, :], 5))
        print('(Test) 95th percentile Value:', np.percentile(self.m.vot_test[modelNumber, :], 95))
        print("Model ", modelNumber, " VOT (test mean): ", np.mean(self.m.vot_test[modelNumber, :]))
        print("Model ", modelNumber, " VOT (test median): ", np.median(self.m.vot_test[modelNumber, :]))
        #ax.set_xlim([np.percentile(temp, 5), np.percentile(temp, 95)])
        #ax.set_xlim([-200, 200])
        ax.set_xlabel("Value of Time (Drive) ("+currency+"/h)")
        ax.set_ylabel('Density')
        fig.savefig('plots/' + self.m.run_dir + '_' + str(modelNumber) + '_vot_drive_train.png', bbox_inches="tight")

    def plot_vot(self, mode = 'drive', currency = '$'):
        """Plotting VOT histogram

        Args:
            mode (str): str of mode. 'pt' or 'drive'. Default at 'drive'
            currency (str): str of currency. e.g. '$'. Default at '$'
        """
        # assert mode should be 'pt' or 'drive'
        assert(mode is 'pt' or mode is 'drive')
        if mode is 'pt':
            # avg_vot = np.nanmean(self.m.vot_pt_test, axis=1)
            avg_vot = np.ma.masked_invalid(self.m.vot_pt_test).mean(axis=1)
        else:
            avg_vot = np.ma.masked_invalid(self.m.vot_drive_test).mean(axis=1)
            # avg_vot = np.nanmean(self.m.vot_drive_test, axis=1)

        # avg_vot = np.mean(self.m.vot_test, axis=1)
        print('VOT: ', avg_vot)
        # exclude nan values in avg
        avg_vot = avg_vot[~np.isnan(avg_vot)]
        print('Dropped ', self.m.numModels - len(avg_vot), ' Models.')
        print('Mean VOT test:', np.mean(avg_vot))
        print('Median VOT test:', np.median(avg_vot))
        print('VOT: ', avg_vot)
        fig, ax = plt.subplots(figsize=(8, 8))
        bins = np.linspace(np.percentile(avg_vot, 5), np.percentile(avg_vot, 95),
                        int((np.percentile(avg_vot, 95) - np.percentile(avg_vot, 5)) / 3))
        ax.hist(avg_vot, bins=bins, density=True, color='b')
        ax.set_xlim([np.percentile(avg_vot, 5), np.percentile(avg_vot, 95)])
        #ax.set_xlim([-200, 200])
        # ax.set_xlabel("Value of Time (Drive) ("+currency+"/h)")
        ax.set_xlabel("Value of Time ({}) ({})/h)".format(mode, currency))
        ax.set_ylabel('Density')
        self.save_plot(fig, '{}_vot_hist'.format(mode))

        # avg_vot_train = np.mean(self.m.vot_train, axis=1)
        # avg_vot_train = avg_vot_train[~np.isnan(avg_vot_train)]
        # print('Dropped ', self.m.numModels - len(avg_vot_train), ' Models.')
        # print('Mean VOT train:', np.mean(avg_vot_train))
        # print('Median VOT train:', np.median(avg_vot_train))
        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.hist(avg_vot_train, bins=bins, density=True, color='b')
        # ax.set_xlim([np.percentile(avg_vot, 5), np.percentile(avg_vot, 95)])
        # #ax.set_xlim([-200, 200])
        # ax.set_xlabel("Value of Time (Drive) ("+currency+"/h)")
        # ax.set_ylabel('Density')
        # fig.savefig('plots/' + self.m.run_dir + '_vot_drive_train.png', bbox_inches="tight")

    def plot_accuracy_histogram(self, m):
        None

if __name__ == "__main__":
    run_dir = 'ldn_models/model15_7000'
    # run_dir = 'sgp_models/model21'
    plt.rcParams.update({'font.size': 24})
    #currency = "£"
    currency = "$"
    modelNum = 10
    dir_plots = 'Plots'
    # load the results from .pkl. This pickle file is generated from model training
    with open(run_dir + '.pkl', 'rb') as f:
        m = pickle.load(f)

    '''
    plot_variables = ['bus_cost ($)', 'bus_ivt (min)', 'rideshare_cost ($)', 'rideshare_ivt (min)', \
                    'av_cost ($)', 'av_ivt (min)', 'drive_cost ($)', 'drive_ivt (min)']
    plot_vars = [1, 4, 5, 7, 8, 10, 11, 13]
    '''
    plot_variables = ['drive_cost ($)']
    plot_vars_standard = [10] # index of standard vars
    plot_vars = [13]
    plot_modes = ['drive']

    assert len(plot_variables) == len(plot_vars) == len(plot_vars_standard) == len(plot_modes)
    plot_NN = Plot_Neural_Net(run_dir + '.pkl', dir_plots, 'original', 'DNN')
    plot_NN.plot_vot(currency)
    
    plot_NN.plot_choice_prob(plot_modes, plot_vars, plot_vars_standard, plot_variables)
    plot_NN.plot_prob_derivative(plot_modes, plot_vars, plot_vars_standard, plot_variables)#, [86,17,31], ['C1: 0.602', 'C2: 0.595', 'C3: 0.586'])
    plot_NN.plot_substitute_pattern(plot_vars, plot_vars_standard, plot_variables)

    # plot_vot(m, currency)
    
    # plot_choice_prob(m, plot_modes, plot_vars, plot_vars_standard, plot_variables)
    # plot_prob_derivative(m, plot_modes, plot_vars, plot_vars_standard, plot_variables)#, [86,17,31], ['C1: 0.602', 'C2: 0.595', 'C3: 0.586'])
    # plot_substitute_pattern(m, plot_vars, plot_vars_standard, plot_variables)
    # plot_onemodel_prob_derivative(m, 78, plot_vars, plot_vars_standard, plot_variables)
    # plot_onemodel_vot(m, 0, currency)

    '''
    # average elasticity of invididuals
    for i in np.arange(len(m.cont_vars)):
        print(m.all_variables[m.cont_vars[i]], end=' & ')
        for j in range(4):
            print("%.3f(%.1f)" % (np.mean(m.elasticity_test[i, 78, j, :]), np.std(m.elasticity_test[i, 78, j, :])), end = ' & ')
            #print(np.mean(m.elasticity[i, 1, 3, :]), np.std(m.elasticity[i, 1, 3, :]))
        print("\n")
        
    # elasticity of average individual
    for i, j in zip([0, 1, 4, 5, 7, 8, 10, 11, 13], 
                    [10, 711, 210, 28, 178, 11, 4, 1234, 506]):
        pb = m.mkt_prob_derivative[i,:,3,j]
        ch = m.mkt_choice_prob[i,:,3, j]
        x = m.X_train_raw[j, i]
        std = m.std[i]
        elas = pb * x / ch / std
        print(np.mean(elas), np.std(elas))\
    '''
    print(np.mean(m.mkt_share_test, axis = 0))
    print(np.std(m.mkt_share_test, axis = 0))

    '''
    print(np.mean(m.sw_change_1))
    print(np.mean(m.sw_change_2))
    print(np.mean(m.sw_change_3))
    '''

    # testing the sw_ind and sw_change_*
    print('Inspecting shape of welfare')
    print(m.sw_ind.shape)
    print(m.sw_change_0.shape)
    print(m.sw_change_1.shape)
    print(m.sw_change_2.shape)
    print(m.sw_change_3.shape)

    print("Overall welfare change")
    print(np.sum(np.mean(m.sw_ind, axis=0)))
    print(np.sum(np.mean(m.sw_change_0, axis=0)))
    print(np.sum(np.mean(m.sw_change_1, axis=0)))
    print(np.sum(np.mean(m.sw_change_2, axis=0)))
    print(np.sum(np.mean(m.sw_change_3, axis=0)))
    # print(np.mean(m.sw_change_0, axis=0))
    # print(np.mean(m.sw_change_1, axis=0))
    # print(np.mean(m.sw_change_2, axis=0))
    # print(np.mean(m.sw_change_3, axis=0))

    print('Number of test set')
    print(m.X_test_standard.shape)