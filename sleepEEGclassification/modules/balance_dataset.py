# coding: utf-8
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class input_output:
    
    def read_dataset_list(path, files = "", header=None):
        '''
        Read and concat a list of datasets
        
        inputs: 
            1. path: if files = "", path is a list with the whole path
            to the files, including the file name. Otherwise is only
            the path to the datasets folder.
            2. files: A list with the datasets names.
            3. header:int, list of int, 'infer', default None
            
            returns: Dataset composed by all of the list
        '''
        
        df_list = []
        
        if files == "":
            files_path = path
        else:
            files_path = [os.path.join(path, x) for x in files]
        
        for file in files_path:
            df_list.append(pd.read_csv(file, header = None))
        
        return pd.concat(df_list, ignore_index = True)           
    
                
    def balance_dataset(df, data_columns, label_column, test_size):
        '''
        Balance classes and split between train and test a pandas
        dataframe. All the classes will have the same amount of the
        class with less samples.
        
        inputs: 
            1. df: pandas dataframe
            2. data_columns (list): columns indexes with the features
            3. label_column (int): column index with the label
            4. test_size (float): percent size of the test set
        
        returns:
            X_train, y_train, X_test, y_test
        '''
        
        classes = df[label_column].unique()

        # Split by class
        df_splits = []
        for cl in classes:
            df_splits.append(df.drop(df.index[df[label_column] == cl]))

        minor_class_len = min([x.shape[0] for x in df_splits])

        df_splits_balanced = []
        
        # Discover which is the class with less samples
        for split in df_splits:
            df_splits_balanced.append(split.sample(n = minor_class_len))
    
        df_splits = df_splits_balanced

        df_train = []
        df_test  = []
    
        # Split between train and test
        for split in df_splits:
            t1, t2 = train_test_split(split, test_size = test_size, shuffle = True)
            df_train.append(t1)
            df_test.append(t2)
        
        test  = pd.concat(df_test,   ignore_index = True)
        train = pd.concat(df_train, ignore_index = True)
        
        test  = test.sample(frac=1).reset_index(drop=True)
        train = train.sample(frac=1).reset_index(drop=True)

        return train[data_columns], train[label_column], test[data_columns], test[label_column]
