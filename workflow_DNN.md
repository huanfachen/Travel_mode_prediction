```nomnoml
#.action: fill=#8c6bb1
#.data: fill=#8c96c6

//[CSV_data]->[Pickle_data]
//[Pickle_data]->[TF_model_file]
//[TF_model_file]->[CSV_result_file]
//[Pickle_data]->[Pickle_model_file]
//[Pickle_model_file]->[PNG_plots]

[<action>preprocess]->[<action>model_train]
[model_train]->[<action>analysis]
[model_train]->[<action>plot]

[CSV_data]--:>[preprocess]
[preprocess]--:>[<data>Pickle_data]
[Pickle_data]--:>[model_train]
[model_train]--:>[<data>TF_model_file]
[model_train]--:>[<data>Pickle_model_file]

[TF_model_file]--:>[analysis]
[analysis]--:>[<data>CSV_result_file]

[Pickle_model_file]--:>[plot]
[plot]--:>[<data>PNG_plots]

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

