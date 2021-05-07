import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

if __name__ == '__main__':
    # to calculate the mean of a list of arrays (of the same shape)
    list_array = list(map(lambda x:np.random.rand(3,2), range(2)))
    print(list_array[0])
    print(np.mean(list_array, axis = 0))

    # np 0-d array, transformed to DataFrame
    test_array = np.array([[1]])
    # test_array = np.array([1])
    print(test_array)
    print(test_array.shape)
    test_df = pd.DataFrame(data=test_array, index = ['row1'], columns = ['col1'])
    print(test_df)

    y_true = [0, 1, 2, 2, 2]
    y_pred = [0, 0, 2, 2, 1]
    target_names = ['class 0', 'class 1', 'class 2']
    test_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    print(type(test_report))
    # a list of accuracy, macro_f1_score, macro_recall
    print([test_report['accuracy'], test_report['macro avg']['f1-score'], test_report['macro avg']['recall']])
    temp_df = pd.DataFrame.from_dict({k: test_report[k] for k in target_names})
    list_class_metrics = [temp_df, temp_df]
    avg_df = np.mean(list_class_metrics, axis = 0)
    
    print(avg_df)
    print(pd.DataFrame(avg_df, index = temp_df.index, columns = temp_df.columns))
    print(pd.DataFrame(temp_df, index = temp_df.index, columns = temp_df.columns))

    print(np.ptp(y_true))
    