import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence
from skrub import Cleaner, TableReport

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

columns_to_drop = [
    'nomi', 'ssri_attuale_dic', 'snri_attuale_dic', 'altriantidep_attuale_dic',
    'ssri_att', 'snri_att', 'altriantidep_att', 'antipsy_att', 'bdz_att', 'moodstab_att', 
    'altreterapie_att', 'ssri_prev', 'snri_prev', 'altriantidep_prev', 'antipsy_prev', 
    'bdz_prev', 'moodstab_prev', 'altro_prev', 'bdz_sost', 'coc_sost', 
    'alcol_sost', 'nicot_sost', 'thc_sost', 'opi_sost', 'amph_sost', 'nps_sost',
    'patologieorganichecomorbili', 'tmta_t0', 'tmtb_t0', 'tmta_t1', 'tmtb_t1',
    'tmta_t2', 'tmtb_t2', 'tmta_t3', 'tmtb_t3', 
    'ctqea_cutoff', 'ctqpa_cutoff', 'ctqen_cutoff', 'ctqpn_cutoff', 'ctqsa_cutoff', 'ctqde_cutoff', 
    'deltat0t1', 'deltat0t2', 'deltat0t2_dicotomico'
]


def get_remodula_data(madrs_cutoff=0.5, percentage_missing_columns=0.45, percentage_missing_subjects=0.45):
    
    path = "/media/robbis/DATA/meg/remodula/"
    fname = "remodula.xlsx"
    
    raw_dataframe = pd.read_excel(os.path.join(path, fname))

    # Change column names to remove spaces, dots, and uppercase
    raw_dataframe.columns = raw_dataframe.columns.str.replace(r'[.\n\s]+', '_', regex=True).str.lower()

    dataframe = raw_dataframe.drop(columns=columns_to_drop)

    # Exclude rows using column 'escludere' and then drop the column
    dataframe = dataframe[dataframe['escludere'] != 1].drop(columns=['escludere'])
    dataframe['madrs_t1'] = dataframe['madrs_t1'].fillna(0)


    # Select columns with t0, t1 and count missing values on a row basis
    t0_columns = [col for col in dataframe.columns if col.endswith('_t0')]
    t1_columns = [col for col in dataframe.columns if col.endswith('_t1')]
    t2_columns = [col for col in dataframe.columns if col.endswith('_t2')]
    t3_columns = [col for col in dataframe.columns if col.endswith('_t3')]

    # Remove madrs_t1 from t1_columns
    t1_columns.remove('madrs_t1')

    num_missing_t0 = dataframe[t0_columns].isnull().sum(axis=1) / len(t0_columns)
    num_missing_t1 = dataframe[t1_columns].isnull().sum(axis=1) / len(t1_columns)

    # Exclude t1, t2, t3 columns
    dataframe = dataframe.drop(columns=t1_columns + t2_columns + t3_columns)

    # Fill missing values in madrs_t1 with zero
    dataframe = dataframe[~np.isnan(dataframe['madrs_t0'])]

    # Clean the dataframe using skrub
    cleaner = Cleaner()
    cleaned_dataframe = cleaner.fit_transform(dataframe)

    # How many missing values are there in each column?
    missing_values = cleaned_dataframe.isnull().sum()
    percentage_missing = missing_values / len(cleaned_dataframe)
    columns_with_too_many_missing = percentage_missing[percentage_missing > percentage_missing_columns]
    names_of_missing = columns_with_too_many_missing.index.tolist()
    logger.info(f"Columns with more than {percentage_missing_columns * 100}% missing values: {names_of_missing}")
    columns_with_too_many_missing = columns_with_too_many_missing.index
    cleaned_dataframe = cleaned_dataframe.drop(columns=columns_with_too_many_missing)

    # How many rows have missing values in each row?
    missing_rows = cleaned_dataframe.isnull().sum(axis=1)
    percentage_missing_rows = missing_rows / len(cleaned_dataframe.T)
    subjects_with_missing = percentage_missing_rows > percentage_missing_subjects
    subjects_codes_with_missing = cleaned_dataframe[subjects_with_missing]['codicepaziente'].values
    logger.info(f"Subjects with more than {percentage_missing_subjects * 100}% missing values: {subjects_codes_with_missing}")

    cleaned_dataframe = cleaned_dataframe[~subjects_with_missing]

    X = cleaned_dataframe.drop(columns=['codicepaziente', 'madrs_t1', 'madrs_t0'])
    y = (cleaned_dataframe['madrs_t1'] / cleaned_dataframe['madrs_t0']) <= madrs_cutoff

    return X, y