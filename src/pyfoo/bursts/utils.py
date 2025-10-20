from pyfoo.bursts import channel2lobes
from mne.io.itab import read_raw_itab
import pandas as pd
import numpy as np
import os
import h5py
from scipy.stats import mannwhitneyu, ttest_ind
from pyitab.results import filter_dataframe


def load_bursts(bands=['alpha', 'beta', 'gamma']):
    datapath =  "/home/robbis/mount/meg_analysis/PDM/"

    # datapath = "F:\\PDM\\"

    # sample_raw = os.path.join(datapath, "ANTEA", "SUBJECTS", "bsllsn94_01", "bsllsn94_0104.raw")
    # raw = read_raw_itab(sample_raw)

    fn_pattern = os.path.join(datapath, "bids", "derivatives", "bursts", 
                              "{subject}", "{subject}_space-sensor_band-{band}_bursts.mat")
    
    cond_pattern = os.path.join(datapath, "bids", "derivatives", "bursts",
                                "{subject}", "{subject}_space-sensor_band-alpha_conditions.mat")

    subjects = os.listdir(os.path.join(datapath, "bids", "derivatives", "bursts"))
    subjects.sort()
    
    dataframe = []
    for band in bands:
        for s, subject in enumerate(subjects):
            subject_df = load_dataframe(fn_pattern, subject, band, cond_pattern)
            dataframe.append(subject_df)

    dataframe = pd.concat(dataframe, ignore_index=True)
    
    for key in ['maximatiming', 'offsettiming', 'onsettiming']:
        dataframe[key] -= 2.15

    # Here, we cast to int     
    for key in ['classLabels', 'trialind', 'channel']:
        dataframe = dataframe.astype({key: 'int32'})

    # We create another variable
    dataframe['level'] = (dataframe['classLabels'] - 1) % 3
    dataframe['response'] = (dataframe['classLabels'] - 1) // 3
    
    dataframe['areas'] = [channel2lobes(ch) for ch in dataframe['channel'].values]
    
    levels = {
        0: 'easy',
        1: 'medium',
        2: 'hard'
    }
    
    level_vector = [levels[l] for l in dataframe['level'].values]
    dataframe['level'] = level_vector

    responses = {
        0: 'hit',
        1: 'miss'
    }

    response_vector = [responses[l] for l in dataframe['response'].values]
    dataframe['response'] = response_vector

    timing = {
        0: 'post',
        1: 'pre'
    }

    t_mask = np.int_(dataframe['onsettiming'] < 0)
    timing_vector = [timing[l] for l in t_mask]
    dataframe['timing'] = timing_vector

    
    return dataframe
    


def load_dataframe(pattern, subject, band, cond_pattern):
    item = dict()
    
    dataframe = []
    bursts_fn = pattern.format(subject=subject, band=band)
    mat = h5py.File(bursts_fn, 'r')
    data = mat['burst_save']

    cond_fn = cond_pattern.format(subject=subject)
    cond_mat = h5py.File(cond_fn, 'r')
    rt = cond_mat['total_reaction_times'][()][0]
    
    for ch in range(data.shape[0]):
        ref = data[ch, 0]
        mat_struct = mat[ref]

        for k in mat_struct['Events']['Events'].keys():

            variable = mat_struct['Events']['Events'][k][()][0]

            item.update({
                k: variable
            })
            
            if k == 'trialind':
                item.update({
                    'response_times': [rt[int(trial) - 1] for trial in variable]
                })

        item.update({
            'channel': np.ones_like(variable, dtype=int) * (ch + 1),
            'subject': [subject for _ in range(variable.shape[0])],
            'band' : [band for _ in range(variable.shape[0])]
        })

        dataframe.append(pd.DataFrame(item.copy()))
        
    return pd.concat(dataframe, ignore_index=True)



def get_statistics(dataframe, field_name, test_variable, bands=['alpha', 'beta', 'gamma']):
    
    areas = np.unique(dataframe['areas'])
    
    results = list()
    for timing in np.unique(dataframe['timing']):
        for band in bands:
            for area in np.unique(areas):
                
                series = []
                for response in ['hit', 'miss']:
                    serie = filter_dataframe(dataframe, 
                                            areas=[area], 
                                            band=[band],
                                            timing=[timing],
                                            response=[response]
                                            )
                    series.append(serie[field_name])
                    
                
                t, p = mannwhitneyu(series[1], series[0])
                
                significative = '-'
                if 0.01 < p < 0.05:
                    significative = '*'
                elif p < 0.01:
                    significative = '**'
                elif p > 0.05:
                    significative = '-'
                
                
                results.append(
                    {
                        'band': band,
                        'area': area,
                        'timing': timing,
                        't': t,
                        'p': p,
                        's': significative
                    }
                )
                
    return pd.DataFrame(results)



def get_statistics_full(dataframe, field_name, test_variable):
    
    areas = np.unique(dataframe['areas'])
    
    results = list()

    for band in ['alpha', 'beta']:
        for area in np.unique(areas):
            
            series = []
            for response in ['hit', 'miss']:
                serie = filter_dataframe(dataframe, 
                                        areas=[area], 
                                        band=[band],
                                        #timing=[timing],
                                        response=[response]
                                        )
                series.append(serie[field_name])
                
            
            t, p = mannwhitneyu(series[1], series[0])
            
            significative = '-'
            if 0.01 < p < 0.05:
                significative = '*'
            elif p < 0.01:
                significative = '**'
            elif p > 0.05:
                significative = '-'
                
            results.append(
                {
                    'band': band,
                    'area': area,
                    #'timing': timing,
                    't': t,
                    'p': p,
                }
            )
                
    return pd.DataFrame(results)