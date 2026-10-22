
import pandas as pd
from sekupy.io.subjects import load_subject_file
import numpy as np
from sekupy.results import apply_function
from nilearn.datasets import fetch_atlas_schaefer_2018
import os

def load_controllability_dataset(year='2024', key='average_controllability'):
    
    if year == '2024':
        # Load the dataset for 2024
        results_fc = pd.read_pickle('/media/robbis/DATA/fmri/monks/results_stability_controllability.pkl')
    elif year == '2025':
        # Load the dataset for 2025
        subject_file = '/media/robbis/DATA/fmri/monks/subjects.csv'

        subjects, extra_sa = load_subject_file(subject_file)
        extra_sa = {k: extra_sa[k][-4:] for k in extra_sa.keys()}


        output_folder = '/media/robbis/DATA/fmri/monks/derivatives/pipeline-nct+july25/'
        results_energy = []
        # Load the results
        for subject in subjects:
            
            subj_df = pd.read_pickle(os.path.join(output_folder, f'sub-{subject}_connectivity_results_july25.pkl'))
            
            results_energy.append(subj_df)
            
        results_fc = pd.concat(results_energy, ignore_index=True)
    else:
        raise ValueError("Year must be either '2024' or '2025'")
    
    df = apply_function(
        results_fc,
        keys=['condition', 'group', 'expertise', 'subject', 'age'],
        attr=key,
        fx=lambda x: np.stack(x).mean(axis=0)
    )

    labels = fetch_atlas_schaefer_2018(n_rois=300, resolution_mm=2)['labels']
    labels = [n.decode('utf-8') for n in labels]
    side = [l.split('_')[1] for l in labels]
    network = [l.split('_')[2] for l in labels]
    area = [l.split('_')[3] for l in labels]


    df = df.explode(key)

    df['node'] = np.hstack([labels] * 60)
    df['side'] = np.hstack([side] * 60)
    df['network'] = np.hstack([network] * 60)
    df['area'] = np.hstack([area] * 60)
    
    return df