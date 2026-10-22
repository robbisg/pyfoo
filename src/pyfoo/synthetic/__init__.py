import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

def generate_syntethic_data(dataframe_ttest, max_iterations=100):
    syntethic_data_test = []
    syntethic_data = []
    
    mean_meditators = dataframe_ttest['mean_a'].values
    mean_novices = dataframe_ttest['mean_b'].values
    
    std_meditators = 0.08 / 3
    std_novices = 0.07 / 3
    
    for j, (mean_meditator, mean_novice) in enumerate(zip(mean_meditators, mean_novices)):
        # Generate random samples for meditators and novices

        for _ in range(max_iterations):
        
            meditators = np.random.normal(loc=mean_meditator, scale=std_meditators, size=12) + 0.015
            novices = np.random.normal(loc=mean_novice, scale=std_novices, size=10)
            
            res = ttest_ind(meditators, novices, equal_var=False)
            t = res.statistic
            p = res.pvalue
            dof = len(meditators) + len(novices) - 2
            
            if p < 0.05 / 7: # Bonferroni correction for 20 comparisons
                significant = '3'
            elif p < 0.01:
                significant = '2'
            elif p < 0.05:
                significant = '1'
            else:
                significant = '0'
                
                
            if int(significant) == int(dataframe_ttest.iloc[j]['significant']):
                break
                
                
        syntethic_data_test.append({
            'network': dataframe_ttest.iloc[j]['network'],
            'condition': dataframe_ttest.iloc[j]['condition'],
            't': t,
            'p': p,
            'dof': dof,
            'mean_meditators': meditators.mean(),
            'mean_novices': novices.mean(),
            'std_meditators': meditators.std(),
            'std_novices': novices.std(),
            'significant': int(significant)
        })
        
        
        for i in range(12):
            syntethic_data.append({
                'network': dataframe_ttest.iloc[j]['network'],
                'condition': dataframe_ttest.iloc[j]['condition'],
                'group': 'meditators',
                'value': meditators[i],
                'subject': f'meditator_{i+1}'
            })
        
        for i in range(10):
            syntethic_data.append({
                'network': dataframe_ttest.iloc[j]['network'],
                'condition': dataframe_ttest.iloc[j]['condition'],
                'group': 'novices',
                'value': novices[i],
                'subject': f'novice_{i+1}'
            })  
        
        
    syntethic_data_test = pd.DataFrame(syntethic_data_test)
    syntethic_data = pd.DataFrame(syntethic_data)
    
    return syntethic_data_test, syntethic_data