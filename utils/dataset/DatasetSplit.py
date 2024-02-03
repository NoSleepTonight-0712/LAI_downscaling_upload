import numpy as np

from torch.utils.data import DataLoader, random_split, Dataset, Subset, SubsetRandomSampler
import pandas as pd
from sklearn.model_selection import train_test_split

def splitDatasetByMonth(dataset, train_count, val_count, test_count, random_state=114514):
    train_indies = []
    train_data = []
    train_label = []

    val_indies =[]
    val_data = []
    val_label = []

    test_indies = []
    test_data = []
    test_label = []

    all_data_count = len(dataset)

    data_index = np.arange(all_data_count)
    data_month = data_index % 12 + 1

    df = pd.DataFrame(data=[data_index, data_month], index=['data_index', 'data_month']).T
    y = df.pop('data_month').to_frame()
    X = df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,stratify=y, train_size=train_count, 
        test_size=val_count + test_count, random_state=random_state)

    X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, stratify=y_test, test_size=test_count,
              train_size=val_count, random_state=random_state)
    
    # train_set_idx = [(idx, 1) for idx in X_train['data_index']]
    # test_set_idx = [(idx, 2) for idx in X_test['data_index']]
    # val_set_idx = [(idx, 3) for idx in X_val['data_index']] 
    train_set_idx = X_train['data_index'].to_numpy()
    test_set_idx = X_test['data_index'].to_numpy()
    val_set_idx = X_val['data_index'].to_numpy()
    # all_set_idx = []
    # all_set_idx.extend(train_set_idx)
    # all_set_idx.extend(test_set_idx)
    # all_set_idx.extend(val_set_idx)
    
    # all_set_idx = sorted(all_set_idx)

    # for idx, feature, label in dataset:
    #     set_idx = all_set_idx[idx]
    #     if set_idx[0] == idx:
    #         catagory = set_idx[1]
    #         if catagory == 1:
    #             # to train set
    #             train_indies.append(idx)
    #             train_data.append(feature)
    #             train_label.append(label)
    #         elif catagory == 2:
    #             # to test set
    #             test_indies.append(idx)
    #             test_data.append(feature)
    #             test_label.append(label)
    #         elif catagory == 3:
    #             # to val set
    #             val_indies.append(idx)
    #             val_data.append(feature)
    #             val_label.append(label)
    #     else:
    #         raise Exception('Not matched index expected!')

    train_set = Subset(dataset, train_set_idx)
    val_set = Subset(dataset, val_set_idx)
    test_set = Subset(dataset, test_set_idx)

    return train_set, val_set, test_set

    