import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import random
import datetime
import glob
import os
import re

if __name__ == '__main__':
    # create a dictionary
    dict_result = {}
    # test
    # dict_result['Original_DNN'] = [0.6847011145, -7.87932384, 18.64552617, 0.1664236983, 0.339386493, 0.02259614567, 0.4715921481]    
    # dict_result['Original_XGBoost'] = [0.6847011145, -7.87932384, 18.64552617, 0.1664236983, 0.339386493, 0.02259614567, 0.4715921481]
    # df_result = pd.DataFrame.from_dict(dict_result, orient='index',
    #                    columns=["accuracy","VOT_drive",	"VOT_pt", "market_share_walk", "market_share_pt", "market_share_cycle",	"market_share_drive"])

    # iterate files
    folder_result = "Results"
    list_pattern_filename = ['*MNL*mean*.csv', '*dnn*mean*.csv', '*xgb*mean*.csv']
    for tmp_pattern_filename in list_pattern_filename:
        list_files_one_method = glob.glob(os.path.join(folder_result, tmp_pattern_filename))
        for tmp_file in list_files_one_method:
            # [0] # the entire match
            # [1] the first parenthesized subgroup
            tmp_data_method = re.search("Results/(.*).csv", tmp_file)[1]
            if tmp_data_method.startswith("Original"):
                # pattern: Original_[method].csv
                list_match = re.search("Results/(.*)_(.*)_mean.csv", tmp_file)
                tmp_data = list_match[1]
                tmp_method = list_match[2]
            else:
                # pattern: [data]_[number]_[method].csv
                list_match = re.search("Results/(.*)_(.*)_(.*)_mean.csv", tmp_file)
                tmp_data = list_match[1]
                tmp_method = list_match[3]
            tmp_list_result = []
            # data
            tmp_list_result.append(tmp_data)
            # method
            tmp_list_result.append(tmp_method)
            # read the fileles
            with open(tmp_file) as file:
                lines = file.readlines()
                # accuracy
                tmp_list_result.append(re.search("accuracy,(.*)$", lines[3 - 1])[1])
                # macro_f1_score
                tmp_list_result.append(re.search("macro_f1_score,(.*)$", lines[4 - 1])[1])
                # macro_recall
                tmp_list_result.append(re.search("macro_recall,(.*)$", lines[5 - 1])[1])
                # f1_walk, f1_cycle, f1_pt, f1_drive
                tmp_list_result.extend(re.search("f1-score,(.*),(.*),(.*),(.*)", lines[11 - 1]).group(1,2,3,4))
                # VOT_drive
                tmp_list_result.append(re.search("VOT_drive,(.*)$", lines[58 - 1])[1])
                # VOT_pt
                tmp_list_result.append(re.search("VOT_pt,(.*)$", lines[62 - 1])[1])
                # market_share_walk
                tmp_list_result.append(re.search("walk,(.*)$", lines[51 - 1])[1])
                # market_share_cycle
                tmp_list_result.append(re.search("cycle,(.*)$", lines[52 - 1])[1])
                # market_share_pt
                tmp_list_result.append(re.search("pt,(.*)$", lines[53 - 1])[1])
                # market_share_drive
                tmp_list_result.append(re.search("drive,(.*)$", lines[54 - 1])[1])
                # data
                # list_match = re.search("(.*)_(\d*)_(.*)_mean", "1.3_NeighbourhoodCleaningRule_9_MNL_mean")
            # add results to the dict
            dict_result[tmp_data_method] = tmp_list_result
    # write to file
    df_result = pd.DataFrame.from_dict(dict_result, orient='index',
                       columns=["data", "method", "accuracy", "macro_f1_score", "macro_recall", "f1_walk", "f1_cycle", "f1_pt", "f1_drive", "VOT_drive", "VOT_pt", "market_share_walk", "market_share_cycle", "market_share_pt", "market_share_drive"])
    # change columns to float. Ignore these columns that are not float
    df_result = df_result.apply(pd.to_numeric, errors='ignore')
    # generate two new columns
    df_groupby = df_result.groupby(['data', 'method'], as_index=False)
    df_groupby.mean().to_csv("Results/result_mean_comparison.csv", index=True)
    df_groupby.std().to_csv("Results/result_std_comparison.csv", index=True)
    df_result.to_csv("Results/result_comparison.csv", index=True)
