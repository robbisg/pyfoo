import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
import seaborn as sns
from sklearn.inspection import partial_dependence
import os


def get_feature_importance(cv, data_rem):
    features = []
    
    for estimator in cv['estimator']:
        features.append(estimator.feature_importances_)
        
    avg_features = np.mean(features, 0)
    args = np.argsort(avg_features)[::-1]
    columns = data_rem.columns[args]
    features = np.array(features)[:, args]
    df_features = pd.DataFrame(features, columns=columns) 
    
    return df_features
    
    


def plot_importance(cv, data_rem, n_features=30):

    df_features = get_feature_importance(cv, data_rem)
    columns = df_features.columns
    
    df_plot = df_features.drop(columns=df_features.columns[n_features:])
    df_plot = pd.melt(df_plot)
    fig, ax = pl.subplots(1, 1, figsize=(15, 13))
    sns.barplot(data=df_plot, y='variable', x='value', orient='h', ax=ax)
      
      
        

def get_partial_dependence(cv, data, columns):
       
    importance = dict()
    for estimator in cv['estimator']:
        for feature in columns:
            results = partial_dependence(estimator, data, features=feature)
            if feature not in importance.keys():
                importance[feature] = dict()
                importance[feature]['values'] = []
            else:
                importance[feature]['values'].append(results[0].squeeze())
                importance[feature]['grid'] = results[1][0]
    
    return importance



def plot_direction(cv, data, columns, plot_shape=(5, 5), color='blue'):
    
    if columns.shape[0] > 25:
        columns = columns[:25]
        plot_shape = (5, 5)
    elif columns.shape[0] <= 1:
        plot_shape = (1, 2)
    else:
        cols = np.rint(np.sqrt(columns.shape[0]))
        rows = np.ceil(columns.shape[0] / cols)
        plot_shape = (int(rows), int(cols))
    
    
    importance = get_partial_dependence(cv, data, columns)
    
    fig, axes = pl.subplots(plot_shape[0], 
                            plot_shape[1], 
                            figsize=(15, 15))
    indices = np.nonzero(np.ones(plot_shape))
    
    for i, feature in enumerate(importance.keys()):
        ax = axes[indices[0][i], indices[1][i]]
        feature_importance = np.array(importance[feature]['values'])
        feature_importance = (feature_importance - feature_importance.min()) \
            / (feature_importance.max() - feature_importance.min())
        ax.plot(importance[feature]['grid'], feature_importance.T, 
                alpha=0.03, c=color)
        ax.plot(importance[feature]['grid'], feature_importance.mean(0), c=color)
        ax.set_title(feature)
        
    pl.tight_layout()
    
    return fig, axes



def get_esketamina_data(kind='t1', response='madrs'):
    
    path = "/media/robbis/DATA/meg/esketamina/"
    data = pd.read_csv(os.path.join(path, "database-esketamina-clean-full.csv"))  
    
    if kind == 't1':
        field = 'MADRSTOTT1'
        dropped_columns = ['Numero progressivo paziente', 
                            'Genere', 
                            'FIGLI (Sì/No)', 
                            'OCCUPAZIONE ', 
                            'MADRSTOTT0', 
                            'MADRSTOTT1', 
                            'STATUS (single, sposato, separato)', 
                            'targets', 
                            'response', 
                            'HAMDTOTT0', 
                            'HAMDTOTT1']
        dropped_columns += ['MADRSITEM-%dT1' % (i+1) for i in range(10)]
        dropped_columns += ['HAMDITEM-%dT1' % (i+1) for i in range(21)]
        dropped_columns += ['HAMAITEM-%dT1' % (i+1) for i in range(14)]
        #dropped_columns += ['MADRSITEM-%dT0' % (i+1) for i in range(10)]
        #dropped_columns += ['HAMDITEM-%dT0' % (i+1) for i in range(21)]
        #dropped_columns += ['HAMAITEM-%dT0' % (i+1) for i in range(14)]
        dropped_columns += list(data.columns[121:])
        
        
    elif kind == 't2':
        field = 'MADRSTOTT2'
        dropped_columns = ['Numero progressivo paziente', 
                            'Genere', 
                            'FIGLI (Sì/No)', 
                            'OCCUPAZIONE ', 
                            'MADRSTOTT0',
                            'MADRSTOTT1',
                            'MADRSTOTT2',
                            'STATUS (single, sposato, separato)', 
                            'targets', 
                            'response', 
                            'HAMDTOTT0',
                            'HAMDTOTT1',
                            'HAMDTOTT2',
                            'HAMATOTT2',
                            ]

        dropped_columns += ['MADRSITEM-%dT2' % (i+1) for i in range(10)]
        dropped_columns += ['HAMDITEM-%dT2' % (i+1) for i in range(21)]
        dropped_columns += ['HAMAITEM-%dT2' % (i+1) for i in range(14)]
        dropped_columns += list(data.columns[-26:])
        
        
    elif kind == 't2_t0':
        field = 'MADRSTOTT2'        
        dropped_columns = ['Numero progressivo paziente', 
                            'Genere', 
                            'FIGLI (Sì/No)', 
                            'OCCUPAZIONE ', 
                            'MADRSTOTT0',
                            'MADRSTOTT1',
                            'MADRSTOTT2',
                            'STATUS (single, sposato, separato)', 
                            'targets', 
                            'response', 
                            'HAMDTOTT0',
                            'HAMDTOTT1',
                            'HAMDTOTT2',
                            'HAMATOTT2',
                            ]

        dropped_columns += ['MADRSITEM-%dT2' % (i+1) for i in range(10)]
        dropped_columns += ['HAMDITEM-%dT2' % (i+1) for i in range(21)]
        dropped_columns += ['HAMAITEM-%dT2' % (i+1) for i in range(14)]
        dropped_columns += list(data.columns[-146:])
        
    elif kind == 't2_t1':
        field = 'MADRSTOTT2'    
        dropped_columns = ['Numero progressivo paziente', 
                        'Genere', 
                        'FIGLI (Sì/No)', 
                        'OCCUPAZIONE ', 
                        'MADRSTOTT0',
                        'MADRSTOTT1',
                        'MADRSTOTT2',
                        'STATUS (single, sposato, separato)', 
                        'targets', 
                        'response', 
                        'HAMDTOTT0',
                        'HAMDTOTT1',
                        'HAMDTOTT2',
                        'HAMATOTT2']

        dropped_columns += ['MADRSITEM-%dT2' % (i+1) for i in range(10)]
        dropped_columns += ['HAMDITEM-%dT2' % (i+1) for i in range(21)]
        dropped_columns += ['HAMAITEM-%dT2' % (i+1) for i in range(14)]
        dropped_columns += list(data.columns[49:49+72])
        dropped_columns += list(data.columns[-74:])
        
    mask = np.logical_not(np.isnan(data[field]))
    data = data.loc[mask]
        
    data['response'] = data[field] / data['MADRSTOTT0']
    
    if response == 'madrs':
        data['targets'] = np.int_(data[field] < 10)
    elif response == 'cutoff':
        data['targets'] = np.int_(data['response'] < 0.5)
    elif response == 'median':
        data['targets'] = np.int_(data['response'] < np.median(data['response']))
       
    y = data['targets'].copy()
    data = data.drop(columns=dropped_columns)
    #data = data.fillna(0)
    data = data.fillna(data.min(axis=0))
    
    mask_edm = data['precedenti EDM (n)'] > 30
    data['precedenti EDM (n)'].loc[mask_edm] = np.mean(data['precedenti EDM (n)'].loc[np.logical_not(mask_edm)])
    
    return data, y