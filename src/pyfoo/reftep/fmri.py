import pandas as pd
import numpy as np
import os
from scipy.io import loadmat
import nibabel as ni
import numpy.linalg as npla
from nilearn.maskers import NiftiSpheresMasker, NiftiMasker
import h5py



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