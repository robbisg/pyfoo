import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from pymatreader import read_mat
from pyfoo.reftep.distribution import extract_all_features

RAW_PATH = '/media/robbis/Seagate_Pt1/REFTEP/'
EEG_PATH = '/home/robbis/mount/meg_analysis/REFTEP++/cleaned/{eeg_id}/EEG_EMG_cleaned.mat'


def extract_mep_distribution(sites=['aalto', 'chieti']):

    distributions = []
    for site in sites:

        if site == 'aalto':
            EEG_PATH = '/home/robbis/mount/c2b/reftep_{site}/derivatives/emg/'
            subjects = os.listdir(EEG_PATH.format(site=site))
            subjects.sort()
            subjects = [s for s in subjects if s.startswith('sub-')]
            subjects = [s for s in subjects if not s.endswith('html')]
            subjects = pd.DataFrame(subjects)
            
        else:
            EEG_PATH = '/home/robbis/mount/meg_analysis/REFTEP++/cleaned/{eeg_id}/EEG_EMG_cleaned.mat'
            subjects = pd.read_csv(os.path.join(RAW_PATH, "reftep.csv"))
        
        for s, subject in subjects.iterrows():

            print("Processing ", subject)
            
            if site == 'aalto':
                subject = subject.values[0]
                bids_id = subject  # f'sub-{bids_subject_id}'
            else:
                eeg_id = subject['EEG'].lower()
                bids_id = subject['BIDS']
                bids_id = 'sub-'+str(bids_id)

            
            if site == 'aalto':
                mep_dir = os.path.join(EEG_PATH.format(site=site), bids_id)

                mep_data = read_mat(os.path.join(mep_dir, 'AmpsM.mat'))
                meps = mep_data['AmpsM']['AmpsM']
                bad_trials_emg = mep_data['badEMGtrials'][0]
                bad_trials_emg_idx = np.where(bad_trials_emg == 1)[0]
            else:
                eeg_id = subject['EEG'].lower()
                mep_data = read_mat(EEG_PATH.format(eeg_id=eeg_id))
                meps = mep_data['AmpsM']
                bad_trials_emg = mep_data['badTrEMG'][0]
                bad_trials_emg_idx = np.where(bad_trials_emg == 1)[0]

            log_mep = np.log10(meps[:,0])
            log_mep[log_mep <= 0] = 0.0001
            
            distributions.append({
                'subject': bids_id,
                'site': site,
                'log_mep': log_mep,
            })

    return distributions