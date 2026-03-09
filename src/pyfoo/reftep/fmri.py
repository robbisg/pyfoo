import pandas as pd
import numpy as np
import os
from scipy.io import loadmat
import nibabel as ni
import numpy.linalg as npla
from nilearn.maskers import NiftiSpheresMasker, NiftiMasker
import h5py
from nilearn import datasets

from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def mni152_to_fsaverage(coords_mni):
    """Convert MNI152 coordinates to fsaverage surface coordinates."""
    mni_to_fsaverage = [
        [ 1.0022,  0.0071, -0.0177,  0.0528],
        [-0.0146,  0.9990,  0.0027, -1.5519],
        [ 0.0129,  0.0094,  1.0027, -1.2012],
        [ 0.0000,  0.0000,  0.0000,  1.0000]
    ]
    mni_to_fsaverage = np.array(mni_to_fsaverage)
    
    n_coords = coords_mni.shape[0]
    coords_mni_homogeneous = np.hstack([coords_mni, np.ones((n_coords, 1))])
    
    coords_fsaverage_homogeneous = coords_mni_homogeneous @ mni_to_fsaverage.T
    coords_fsaverage = coords_fsaverage_homogeneous[:, :3]
    
    return coords_fsaverage



def get_schaefer_mask(network_name, n_networks=7, scale=400):
    """Get mask for parcels belonging to a specific Schaefer network."""
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=scale, 
        yeo_networks=n_networks,
        resolution_mm=2
    )
    
    label_mask = np.array([label.decode('utf-8').split('_')[2] for label in atlas.labels])
    label_mask = [network_name in label for label in label_mask]
    label_colors = np.nonzero(label_mask)[0] + 1

    atlas_map = ni.load(atlas.maps)
    atlas_data = atlas_map.get_fdata()
    atlas_mask = np.isin(atlas_data, label_colors, assume_unique=False)

    network_mask = ni.Nifti1Image(atlas_mask.astype(np.int32), atlas_map.affine, atlas_map.header)

    return network_mask, label_colors


def get_affine(nifti):

    voxToWorldMat = nifti.header.get_best_affine()
    shape = nifti.shape
    pixdim = nifti.header.get_zooms()
    
    voxToScaledVoxMat = np.diag(list(pixdim) + [1.0])
    isneuro = npla.det(voxToWorldMat) > 0

    if isneuro:
        x = (shape[0] - 1) * pixdim[0]
        
        flip = np.eye(4)
        flip[0, 0] = -1
        flip[0, 3] = x
        
        voxToScaledVoxMat = flip @ voxToScaledVoxMat
                
    return voxToScaledVoxMat



def load_mep_data(path, subject, fname):
    mep_data = loadmat(os.path.join(path.format(subject=subject), fname))

    meps = mep_data['AmpsM']
    bad_trials_emg = mep_data['badTrEMG'][0]
    bad_trials_emg_idx = np.where(bad_trials_emg == 1)[0]
    
    return meps, bad_trials_emg_idx


def load_nexstim_data(nexstim_path, bids_subject_id, session_file, tmsmri_fname):
    return

def coregistration_nexstim_mni():
    mni_template = '/home/robbis/Downloads/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz'


    tms2fmri_mat = '/home/robbis/sub-{subject}_desc-tranforms_tms2fmri.mat'.format(subject=bids_subject_id)
    tms2fmri_data = '/home/robbis/sub-{subject}_desc-tms2fmri_T1w.nii.gz'.format(subject=bids_subject_id)

    command = "flirt -in {mri_data} -ref {mni_template} -omat {tms2fmri_mat} -out {tms2fmri_data} -coarsesearch 65 -usesqform"

    exe_command = command.format(mri_data=tmsmri_fname,
                                 mni_template=mni_template,
                                 tms2fmri_mat=tms2fmri_mat,
                                 tms2fmri_data=tms2fmri_data)

    print(exe_command)
    
    return


def get_nexstim2mni_matrix(orig_fname, coregistered_fname, coregistered_mat):
    
    ###############################################################################
    # Source Affine
    src = ni.load(orig_fname)
    premat = get_affine(src)
    premat = npla.inv(premat)

    ###############################################################################
    # Reference Affine
    ref = ni.load(coregistered_fname)
    postmat = get_affine(ref)
    
    #######################################################################
    # TMS space to MRI space
    
    orientation = ni.aff2axcodes(src.affine)
    orientation = ''.join(orientation)   
    
    translation = src.shape
    
    if orientation == 'PIR':
        tms2mri = np.array([[ 0,  0, -1, translation[0]],
                            [ 0, -1,  0, translation[1]],
                            [-1,  0,  0, translation[2]],
                            [ 0,  0,  0, 1]])
    else:
        tms2mri = np.array([[-1, 0, 0, translation[0]],
                            [ 0, 0, 1, 0],
                            [ 0, 1, 0, 0],
                            [ 0, 0, 0, 1]])
        
        
    ###############################################################################
    # MRI to MNI
    if coregistered_mat.endswith('.mat'):
        tms2mni_mat = np.loadtxt(coregistered_mat)
    else: # ANT file with extension .h5
        mat = h5py.File(coregistered_mat)
        tms2mni_mat = mat['TransformGroup/1/TransformParameters'][:]
        matrix = tms2mni_mat[:9].reshape(3, 3)
        offset = tms2mni_mat[9:12][:, np.newaxis]
        tms2mni_mat = np.hstack([matrix, offset])
        tms2mni_mat = np.vstack([tms2mni_mat, [0, 0, 0, 1]]).reshape(4, 4)
        #tms2mni_mat = tms2mni_mat.T
    
    
    tms2mni_xyz_matrix = (postmat @ (tms2mni_mat @ (premat @ tms2mri)))
    
    return tms2mni_xyz_matrix, postmat, premat, tms2mri, tms2mni_mat
    



def extract_confounds(confound_tsv, confounds, dt=True):
    '''
    Arguments:
        confound_tsv                    Full path to confounds.tsv
        confounds                       A list of confounder variables to extract
        dt                              Compute temporal derivatives [default = True]
        
    Outputs:
        confound_mat                    
    '''
    
    if dt:    
        dt_names = ['{}_derivative1'.format(c) for c in confounds]
        confounds = confounds + dt_names
    
    # Load in data using Pandas then extract relevant columns
    confound_df = pd.read_csv(confound_tsv, delimiter='\t') 
    confound_df = confound_df[confounds]
    
 
    # Convert into a matrix of values (timepoints)x(variable)
    confound_mat = confound_df.values 
    
    confound_mat = np.nan_to_num(confound_mat)
    
    # Return confound matrix
    return confound_mat


def seed_connectivity(seed, func_filename, confound_filename, confound_variables):
    
    
    bids_subject = 'sub-201'

    derivatives_path = '/home/robbis/mount/c2b/reftep/derivatives/fmriprep/{subject}/ses-mri1/func/'
    confound_pattern = '{subject}_ses-mri1_task-rest_desc-confounds_timeseries.tsv'
    func_pattern = '{subject}_ses-mri1_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'


    func_filename = os.path.join(derivatives_path.format(subject=bids_subject), 
                                func_pattern.format(subject=bids_subject))
    confound_filename = os.path.join(derivatives_path.format(subject=bids_subject),
                                    confound_pattern.format(subject=bids_subject))

    confound_variables = ['trans_x', 'trans_y', 'trans_z',
                        'rot_x', 'rot_y', 'rot_z', 'global_signal',
                        'white_matter', 'csf']

    confounds = extract_confounds(confound_filename,
                                  confound_variables)
            
    confounds[np.isnan(confounds)] = 0
    
    seed_masker = NiftiSpheresMasker(
        [seed],
        radius=3,
        detrend=True,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        low_pass=0.1,
        high_pass=0.01,
        t_r=.69,
        memory="nilearn_cache",
        memory_level=1,
        verbose=0,
        )

    brain_masker = NiftiMasker(
        smoothing_fwhm=3,
        detrend=True,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        low_pass=0.1,
        high_pass=0.01,
        t_r=.69,
        memory="nilearn_cache",
        memory_level=1,
        verbose=0,
    )

    brain_time_series = brain_masker.fit_transform(
        func_filename, confounds=confounds
    )

    seed_time_series = seed_masker.fit_transform(
        func_filename, confounds=confounds
    )

    seed_to_voxel_correlations = (
        np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0]
    )

    seed_to_voxel_correlations_img = brain_masker.inverse_transform(
        seed_to_voxel_correlations.T
    )
    
    return seed_to_voxel_correlations_img


def load_connectivity_matrix(subject_id, site, base_dir="/home/robbis/mount/c2b/reftep_{site}/derivatives/parcel-connectivity/"):
    """
    Load the connectivity matrix for a given subject and site.
    Returns a numpy array or None if not found.
    """
    conn_dir = Path(base_dir.format(site=site))
    conn_file = conn_dir / f"sub-{subject_id}_connectivity.npy"
    if conn_file.exists():
        return np.load(conn_file)
    else:
        print(f"Connectivity file not found: {conn_file}")
        return None

def load_mep_values(subject_id, mep_csv="/home/robbis/mount/c2b/reftep/derivatives/mep-features/mep_features.csv"):
    """
    Load the MEP values for a given subject from the CSV file.
    Returns a pandas Series or None if not found.
    """
    df = pd.read_csv(mep_csv)
    row = df[df['subject'] == f"sub-{subject_id}"]
    if not row.empty:
        return row.iloc[0]
    else:
        print(f"MEP values not found for subject: sub-{subject_id}")
        return None
    
    
def get_subject_files(subject_id, bids_root, session=None):
    """
    Get fMRI and confounds files for a subject.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    bids_root : Path
        Root directory of BIDS dataset
    session : str, optional
        Session identifier
        
    Returns
    -------
    func_files : list of Path
        List of functional image files
    confounds_files : list of Path
        List of confounds files
    """
    logger.info(f"Getting files for subject {subject_id}")
    
    # Construct file patterns
    if session:
        func_pattern = f"sub-{subject_id}/ses-{session}/func/sub-{subject_id}_ses-{session}_*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        confounds_pattern = f"sub-{subject_id}/ses-{session}/func/sub-{subject_id}_ses-{session}_*_desc-confounds_timeseries.tsv"
    else:
        func_pattern = f"sub-{subject_id}/func/sub-{subject_id}_*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        confounds_pattern = f"sub-{subject_id}/func/sub-{subject_id}_*_desc-confounds_timeseries.tsv"
    
    # Find files
    func_files = list(bids_root.glob(func_pattern))
    confounds_files = list(bids_root.glob(confounds_pattern))
    
    if not func_files:
        logger.warning(f"No functional files found for subject {subject_id}")
    
    if not confounds_files:
        logger.warning(f"No confound files found for subject {subject_id}")
    
    return func_files, confounds_files