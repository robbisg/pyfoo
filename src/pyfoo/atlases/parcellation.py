import nibabel as nib
import numpy as np
import mne
import os

from nilearn.datasets import fetch_atlas_schaefer_2018
from neuromaps import transforms
from neuromaps.datasets import fetch_atlas

def get_schaefers_labels(scale=300):
    parcellation = fetch_atlas_schaefer_2018(n_rois=scale)
    nifti_atlas = nib.load(parcellation['maps'])
    labels = [l.decode() if isinstance(l, bytes) else l for l in parcellation['labels']]
    
    # Transform vertices to fsaverage 164k space
    fsaverage_schaefer = transforms.mni152_to_fsaverage(nifti_atlas, 
                                                        fsavg_density='164k', 
                                                        method='nearest')
    
    atlas = fetch_atlas("fsaverage", density='164k')
    l_vertices = nib.load(atlas["white"].L).agg_data('NIFTI_INTENT_POINTSET')
    r_vertices = nib.load(atlas["white"].R).agg_data('NIFTI_INTENT_POINTSET')
    
    mne_labels = []
    annotation = []
    for s, side in enumerate(['lh', 'rh']):
        vertex_data = fsaverage_schaefer[s].darrays[0].data
        
        positions = l_vertices if side == 'lh' else r_vertices
        
        for l, label in enumerate(labels):
            if side.upper() in label:
                vertices_indices = np.where(vertex_data == l + 1)[0]
                pos = positions[vertices_indices]
                
                label = mne.Label(vertices=vertices_indices, 
                                  pos=pos,
                                  subject='fsaverage',
                                  name=label, 
                                  hemi=side)
                
                mne_labels.append(label)
                
    return mne_labels


def get_schaefers_annotations(scale=300, networks=17):
    
    parcellation = fetch_atlas_schaefer_2018(n_rois=scale, yeo_networks=networks)
    map_fname = parcellation['maps']
    
    # Get the path to labels_fname without the file
    labels_dir = os.path.dirname(map_fname)
    labels_fname = os.path.join(labels_dir, 
                                f'Schaefer2018_{scale}Parcels_{networks}Networks_order.txt')
    
    # Load the labels from the text file    
    labels = np.genfromtxt(labels_fname, dtype=str, delimiter='\t')
    
    # Transform vertices to fsaverage 164k space
    nifti_atlas = nib.load(parcellation['maps'])
    fsaverage_schaefer = transforms.mni152_to_fsaverage(nifti_atlas, 
                                                        fsavg_density='164k', 
                                                        method='nearest')
    
    
    l_vertices = fsaverage_schaefer[0].darrays[0].data  # Assuming left hemisphere for simplicity
    l_ctab = labels[int(scale / 2):, 2:5].astype(int)  # Extract RGB values for left hemisphere labels
    l_names = labels[int(scale / 2):, 1]  # Extract label names for left hemisphere
    
    r_vertices = fsaverage_schaefer[1].darrays[0].data  # Assuming right hemisphere for simplicity
    r_ctab = labels[:int(scale / 2), 2:5].astype(int)  # Extract RGB values for right hemisphere labels
    r_names = labels[:int(scale / 2), 1]  # Extract label names for right hemisphere
    
    l_ctab = np.hstack((l_ctab, np.full((l_ctab.shape[0], 1), 255)))  # Add alpha channel
    r_ctab = np.hstack((r_ctab, np.full((r_ctab.shape[0], 1), 255)))  # Add alpha channel
    
    l_ctab = np.hstack((l_ctab, np.arange(1, l_ctab.shape[0] + 1).reshape(-1, 1)))  # Add label index
    r_ctab = np.hstack((r_ctab, np.arange(1, r_ctab.shape[0] + 1).reshape(-1, 1)))  # Add label index
    
    return (l_vertices, l_ctab, l_names), (r_vertices, r_ctab, r_names)