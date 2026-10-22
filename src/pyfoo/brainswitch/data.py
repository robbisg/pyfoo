import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence
from skrub import Cleaner, TableReport
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

columns_to_drop = []


def get_brainswitch_data(timepoint='1', 
                         predictor='ccq',
                         data_source='final',
                         maximal_timepoint='1',
                         percentage_missing_columns=0.45, 
                         percentage_missing_subjects=0.45):
    
    path = "/home/robbis/data/brainswitch/"

    if data_source == 'final':
        fname = "database_brainswitch_definitivo.xlsx"
    elif data_source == 'firenze':
        fname = "database_firenze_copia.xlsx"
    else:
        raise ValueError(f"data_source must be 'biological' or 'basic', got {data_source}")
        
    raw_dataframe = pd.read_excel(os.path.join(path, fname))

    # Change column names to remove spaces, dots, and uppercase
    raw_dataframe.columns = raw_dataframe.columns.str.replace(r'[.\n\s]+', '_', regex=True).str.lower()
    
    # Remove from column names the things between parentheses and the characters before the parentheses
    raw_dataframe.columns = raw_dataframe.columns.str.replace(r'.\(.*?\)', '', regex=True).str.strip()    
    
    # Remove from column names the underscores if afterwards there is a number
    raw_dataframe.columns = raw_dataframe.columns.str.replace(r'_(?=\d)', '', regex=True)


   # How many missing values are there in each column?
    missing_values = raw_dataframe.isnull().sum()
    percentage_missing = missing_values / len(raw_dataframe)
    columns_with_too_many_missing = percentage_missing[percentage_missing > percentage_missing_columns]
    names_of_missing = columns_with_too_many_missing.index.tolist()
    logger.info(f"Columns with more than {percentage_missing_columns * 100}% missing values: {names_of_missing}")
    columns_with_too_many_missing = columns_with_too_many_missing.index
    cleaned_dataframe = raw_dataframe.drop(columns=columns_with_too_many_missing)

    # Clean the dataframe using skrub
    cleaner = Cleaner()
    cleaned_dataframe = cleaner.fit_transform(cleaned_dataframe)

    timepoint_columns = []
    
    if int(maximal_timepoint) > int(timepoint):
        maximal_timepoint = timepoint + 1        
    
    for i in range(int(maximal_timepoint), 8):
        timepoint_columns += [col for col in cleaned_dataframe.columns if col.endswith(f'{i}')]

    logger.info(f"Removing columns for timepoints greater than {maximal_timepoint}: {timepoint_columns}")
    
    
    timepoint_columns.remove(f'{predictor}{timepoint}')
    cleaned_dataframe = cleaned_dataframe.drop(columns=timepoint_columns)

 
    # How many rows have missing values in each row?
    missing_rows = cleaned_dataframe.isnull().sum(axis=1)
    percentage_missing_rows = missing_rows / len(cleaned_dataframe.T)
    subjects_with_missing = percentage_missing_rows > percentage_missing_subjects
    subjects_codes_with_missing = cleaned_dataframe[subjects_with_missing]['patientnumber'].values
    logger.info(f"Subjects with more than {percentage_missing_subjects * 100}% missing values: {subjects_codes_with_missing}")

    cleaned_dataframe = cleaned_dataframe[~subjects_with_missing]
    
    cleaned_dataframe = cleaned_dataframe[~np.isnan(cleaned_dataframe[f'{predictor}{timepoint}'])]
    
    to_be_removed = ['id', 'patientnumber', f'{predictor}{timepoint}', f'{predictor}0']
    

    y = cleaned_dataframe[f'{predictor}{timepoint}'] < 2
    X = cleaned_dataframe.drop(columns=to_be_removed)
    
    print("Columns in X:", X.columns)
    
    return X, np.int_(y)