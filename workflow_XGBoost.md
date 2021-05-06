```nomnoml
#.action: fill=#8c6bb1
#.data: fill=#8c96c6

//[CSV_data]->[Pickle_data]
//[Pickle_data]->[model_file???]
//[model_file???]->[CSV_result_file]
//[Pickle_data]->[Pickle_model_file]
//[Pickle_model_file]->[PNG_plot_XGBs]

[<action>preprocess]->[<action>model_train_XGB]
[model_train_XGB]->[<action>analysis_XGB]
[model_train_XGB]->[<action>plot_XGB]

[CSV_data]--:>[preprocess]
[preprocess]--:>[<data>Pickle_data]
[Pickle_data]--:>[model_train_XGB]
[model_train_XGB]--:>[<data>model_file???]
[model_train_XGB]--:>[<data>Pickle_model_file]

[model_file???]--:>[analysis_XGB]
[analysis_XGB]--:>[<data>CSV_result_file]

[Pickle_model_file]--:>[plot_XGB]
[plot_XGB]--:>[<data>PNG_plot_XGBs]

[<data>CSV_data|
\[Data\]_X_train.csv
\[Data\]_Y_train.csv
\[Data\]_X_test.csv
\[Data\]_Y_test.csv
]

[<data>Pickle_data|\[Data\]_raw.pkl
\[Data\]_processed.pkl
]

[<data>Pickle_model_file|\[Data\]_\[Method\]_Model.pkl]

```

