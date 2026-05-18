# Return montage from NexstimSession and reordering files.
from mne.channels import read_custom_montage
import mne
import numpy as np

def interpolate_missing_channels_position(epochs):
    # Define the standard montage and adjacency for interpolation
    BVEF_PATH = '/home/robbis/git/BC-TMS-128.bvef'    
    
    standard_montage = read_custom_montage(BVEF_PATH)
    channel_names = standard_montage.ch_names[2:]  # Exclude non-EEG channels
    
    standard_info = mne.create_info(ch_names=channel_names, sfreq=1000.0, ch_types='eeg')
    standard_info.set_montage(standard_montage)
    adjacency, adj_chnames = mne.channels.find_ch_adjacency(standard_info, ch_type='eeg')
    
    montage = epochs.get_montage()
    
    # Identify missing channels
    missing_channels = [ch for ch in channel_names if ch not in montage.ch_names]
    for ch in missing_channels:
        print(f"Channel {ch} is missing from the montage.")
        adjacency_mask = np.isin(adj_chnames, missing_channels)
        adjacency_indices = adjacency.toarray()[adjacency_mask]
        adjacency_channels = np.array(adj_chnames)[adjacency_indices[0].nonzero()]
        print(f"Adjacent channels to {ch}: {adjacency_channels}")
        
        pos = epochs.get_montage().get_positions()['ch_pos']
        
        loc = [pos.get(ch) for ch in adjacency_channels]
        interpolated_loc = np.nanmean(loc, axis=0)
        
        print(f"Interpolated location for {ch}: {interpolated_loc}")
        # You can choose to add the interpolated location to the montage if needed
    
        epochs.info['chs'][epochs.info['ch_names'].index(ch)]['loc'][:6] = np.zeros(6) # Ensure the missing channel is in the info
        epochs.info['chs'][epochs.info['ch_names'].index(ch)]['loc'][:3] = interpolated_loc  # Set the location for the missing channel
        
    return epochs