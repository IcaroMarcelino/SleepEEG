# coding: utf-8
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

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
    
                
    def balance_dataset(df, data_columns, label_column, test_size, ratio = 1):
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
        
        X = df[df.columns[data_columns]].values
        y = df[df.columns[label_column]].values
        
        rus = RandomUnderSampler(sampling_strategy = ratio)
        #rus = SMOTE()
        X, y = rus.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle = True)

        return X_train, y_train, X_test, y_test
