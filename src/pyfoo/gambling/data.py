import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
import seaborn as sns
from sklearn.inspection import partial_dependence
import os


def dummy_columns(dataframe, keyword, mapping):

    # Assuming your data is in a DataFrame called 'dataframe' and the column is named keyword
        
    dataframe[keyword] = dataframe[keyword].astype(str)  # Convert everything to strings for consistency
    dataframe[keyword] = dataframe[keyword].replace('nan', np.nan)  # Replace 'nan' strings with NaN values

    # Split combined categories and create a list of all unique categories
    all_categories = set()
    for categories in dataframe[keyword]:
        if pd.notna(categories):  # Skip NaN values
            for category in categories.split(','):
                all_categories.add(category.strip())  # Strip leading/trailing spaces

    # Create a new DataFrame with dummy variables
    columns = list(mapping.values())
    dataframe_dummies = pd.DataFrame(0, index=dataframe.index, columns=columns)

    # Fill in the dummy variables
    for i, categories in enumerate(dataframe[keyword]):
        if pd.notna(categories):
            for category in categories.split(','):
                dataframe_dummies.at[i, mapping[category.strip()]] = 1

    # Combine the original DataFrame with the dummy variables
    dataframe_final = pd.concat([dataframe, dataframe_dummies], axis=1)
    dataframe_final = dataframe_final.drop(columns=[keyword])
    
    return dataframe_final
         


def get_gambling_data(path="/media/robbis/DATA/meg/gambling/", 
                      fname="gambling.xlsx",
                      field="GSAS_Final",
                      treatment="all",
                      baseline="GSAS_Baseline",
                      strategy=None,
                      dropped_columns=[],
                      responder_percentage=0.35):
        
    #path = "/media/robbis/DATA/meg/gambling/"
    #field = "GSAS_Final"
    #baseline = "GSAS_Baseline"
    #responder_percentage = 0.35

    pharmacological_treatments = np.array([2, 3, 4])
    
    data = pd.read_excel(os.path.join(path, fname))
    data = data.iloc[:-3, :-2]
    data.columns = data.columns.str.replace(r'[.\n\s]+', '_', regex=True)

    default_dropped_columns = [
        'ID', 'STUDY', 'Fam_Hx_Gamble_Full', 'Full_Pers__Drug_Use/abuse',
        #'CGI_Imp_Investigator', 
        #'CGI_Final', 
        #'CGI_Baseline',
        'GSAS_Intermediate', 'GSAS_Early',
        #'GSAS_Baseline',
        'Weight', 'Height'
    ]
    
    default_dropped_columns += [field]
    default_dropped_columns += dropped_columns
    
    # Processing columns with categorical values
    mapping = {
                '0': 'Alc_None',
                '1': 'Alc_Father',
                '2': 'Alc_Mother',
                '3': 'Alc_Sibling',
                '4': 'Alc_Uncle',
                '5': 'Alc_Child',
                }

    data = dummy_columns(data, 'Full_Fam_Hx_Alc_Probs', mapping)
    
    # Marital status
    mapping = {
                '2.0': 'Marital_Married',
                '1.0': 'Marital_Single',
                '3.0': 'Marital_Divorced',
                '4.0': 'Marital_LVGTOG',
                '5.0': 'Marital_Gay',
                '6.0': 'Marital_Widowed',
                '0.0': 'Marital_Other'
                }
    
    data = dummy_columns(data, 'Marital_Status', mapping)
    
    # Race category
    mapping = {
                '1.0': 'Race_Caucasian',
                '2.0': 'Race_AfricanAmerican',
                '3.0': 'Race_Latino',
                '4.0': 'Race_Asian',
                '5.0': 'Race_NativeAmerican',
                '6.0': 'Race_Other'
    }
    
    data = dummy_columns(data, 'Race', mapping)

    mask = np.logical_not(np.isnan(data[field]))
    data = data.loc[mask]

    data['response'] = 1 - data[field] / data[baseline]
    data['targets'] = np.int_(data['response'] <= responder_percentage)

    y = data['targets'].copy()
    data = data.drop(columns=default_dropped_columns + ['targets', 'response'])
    
    if treatment == 'all':
        
        return data, y
    
    elif treatment == 'pharmacological' and strategy is None:
    
        mask = data['Group_Intervention'] != 1
        mask = np.logical_and(mask, data['Group_Intervention'] != 6)
        data = data.loc[mask]
        y = y.loc[mask]
    
    elif treatment in pharmacological_treatments and strategy == 'ovr':
    
        mask = data['Group_Intervention'] != 1
        mask = np.logical_and(mask, data['Group_Intervention'] != 6)
        data = data.loc[mask]
       
        y = y.loc[mask]
        y[data['Group_Intervention'] != treatment] = 1 - y[data['Group_Intervention'] != treatment]
        
    elif treatment == 'pharmacological' and strategy == 'ovo':
        
        mask = data['Group_Intervention'] != 1
        mask = np.logical_and(mask, data['Group_Intervention'] != 6)
        data = data.loc[mask]
        y = y.loc[mask] + 1
        
        y = y + (2*(data['Group_Intervention'] - 2))
        
    else:
    
        mask = data['Group_Intervention'] == treatment
        data = data.loc[mask]
        y = y.loc[mask]

    return data, y