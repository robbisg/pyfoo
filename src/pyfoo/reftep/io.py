import os
import glob
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

_C2B_ROOT = '/home/robbis/mount/c2b'
_MEG_ANALYSIS = '/home/robbis/mount/meg_analysis/REFTEP++'
_RAW_CHIETI = '/media/robbis/Expansion_24T/REFTEP_ORIG'

_CHIETI_CSV = os.path.join(_RAW_CHIETI, 'reftep.csv')


@dataclass
class SubjectFiles:
    """All file paths for a REFTEP subject, organised by modality.

    Attributes that are ``None`` are not applicable for the given site.
    ``tms_raw`` is a list (resolved via glob; may be empty).
    """

    # Identifiers
    site: str
    bids_name: str
    bids_id: str
    fs_subject: str
    eeg_id: Optional[str]
    tms_id: Optional[str]
    bids_root: str
    deriv: str

    # EEG
    eeg_raw: str                    # BIDS raw resting-state EEG (.set)
    eeg_clean: str                  # Clean epochs .fif (BIDS derivatives)
    eeg_prepro_dir: str
    eeg_rs: Optional[str]           # EEGLAB cleaned RS (chieti meg_analysis)
    eeg_emg: Optional[str]          # EEG+EMG .mat (chieti meg_analysis)

    # MEP
    mep_mat: Optional[str]          # MEP amplitudes .mat (aalto only)

    # TMS
    tms_session: Optional[str]      # Nexstim .nbe session file
    tms_raw: List[str]              # Glob-resolved raw .nbe list (chieti only)

    # Anatomical MRI
    mri_anat: str

    # fMRI (fmriprep)
    fmri_func: str
    fmri_confounds: str
    fmri_mask: str
    fmri_func_dir: str

    # TMS-to-MNI coregistration (chieti only)
    coreg_matrix: Optional[str]
    coreg_image: Optional[str]
    coreg_dir: Optional[str]

    # FreeSurfer
    fs_dir: str
    fs_src: str
    fs_trans: str

    # EEG source reconstruction
    source_dir: str

    # Connectivity
    connectivity_dir: str

    # Digitised montage (from alignment_algorithm)
    montage_dir: str
    montage_fname: str

    def __repr__(self):
        lines = [f"SubjectFiles(site={self.site!r}, subject={self.bids_name!r})"]
        groups = {
            'EEG':          ['eeg_raw', 'eeg_clean', 'eeg_rs', 'eeg_emg'],
            'MEP':          ['mep_mat'],
            'TMS':          ['tms_session', 'tms_raw'],
            'MRI':          ['mri_anat'],
            'fMRI':         ['fmri_func', 'fmri_confounds', 'fmri_mask'],
            'Coregistration': ['coreg_matrix', 'coreg_image'],
            'FreeSurfer':   ['fs_dir', 'fs_src', 'fs_trans'],
            'Source/Conn':  ['source_dir', 'connectivity_dir'],
        }
        for group, keys in groups.items():
            lines.append(f"  [{group}]")
            for k in keys:
                v = getattr(self, k)
                lines.append(f"    {k}: {v}")
        return '\n'.join(lines)


def _normalize_subject(subject):
    """Return (bids_name, bids_id): ('sub-201', '201')."""
    s = str(subject)
    if s.startswith('sub-'):
        return s, s[4:]
    return f'sub-{s}', s


def _lookup_chieti_csv(bids_id, csv_path=None):
    """Return the subject row from reftep.csv, or None if not found / missing."""
    path = csv_path or _CHIETI_CSV
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    match = df[df['BIDS'].astype(str) == bids_id]
    return match.iloc[0] if not match.empty else None


def get_subject_files(site, subject, csv_path=None):
    """Return a :class:`SubjectFiles` with all file paths for the given subject.

    Parameters
    ----------
    site : str
        ``'chieti'`` or ``'aalto'``.
    subject : str or int
        BIDS subject ID: numeric (``201``), plain string (``'201'``),
        or prefixed (``'sub-201'``).
    csv_path : str, optional
        Path to ``reftep.csv`` (chieti only). Defaults to the standard location.

    Returns
    -------
    SubjectFiles
    """
    bids_name, bids_id = _normalize_subject(subject)

    bids_root = os.path.join(_C2B_ROOT, f'reftep_{site}')
    deriv = os.path.join(bids_root, 'derivatives')

    space = 'MNI152NLin2009cAsym'

    # ------------------------------------------------------------------
    # Site: chieti
    # ------------------------------------------------------------------
    if site == 'chieti':
        row = _lookup_chieti_csv(bids_id, csv_path)
        eeg_id = row['EEG'].lower() if row is not None else None
        tms_id = row['TMS'] if row is not None else None

        fs_subject = f'{bids_name}_ses-tmseeg'
        fmri_ses = 'ses-mri1'
        ses_tag = f'_{fmri_ses}'

        mri_anat = os.path.join(
            bids_root, bids_name, 'ses-tmseeg', 'anat',
            f'{bids_name}_ses-tmseeg_T1w.nii.gz'
        )
        fmri_func_dir = os.path.join(deriv, 'fmriprep', bids_name, fmri_ses, 'func')

        coreg_dir = os.path.join(deriv, 'tmscoregistration', bids_name)
        coreg_matrix = os.path.join(
            coreg_dir, f'{bids_name}_from-tms_to-mni_desc-flirt_matrix.mat'
        )
        coreg_image = os.path.join(
            coreg_dir, f'{bids_name}_from-tms_to-mni_desc-flirt_T1w.nii.gz'
        )

        tms_session = os.path.join(_RAW_CHIETI, 'sessions', f'session_{bids_id}.nbe')
        if tms_id:
            pattern = os.path.join(
                _RAW_CHIETI, 'sourcedata', 'TMS', f'*{tms_id}*', 'TextExport', '*.nbe'
            )
            tms_raw = glob.glob(pattern)
        else:
            tms_raw = []

        eeg_rs = (os.path.join(_MEG_ANALYSIS, 'cleaned_RS', f'{eeg_id}_RScleaned.set')
                  if eeg_id else None)
        eeg_emg = (os.path.join(_MEG_ANALYSIS, 'cleaned', eeg_id, 'EEG_EMG_cleaned.mat')
                   if eeg_id else None)

        eeg_prepro_dir = os.path.join(deriv, 'eegprepro-rest', bids_name, 'ses-tmseeg', 'eeg')
        eeg_clean = os.path.join(
            eeg_prepro_dir,
            f'{bids_name}_ses-tmseeg_task-rest_run-01_proc-clean_epo.fif'
        )
        mep_mat = None

    # ------------------------------------------------------------------
    # Site: aalto
    # ------------------------------------------------------------------
    elif site == 'aalto':
        eeg_id = None
        tms_id = None

        fs_subject = bids_name  # no _ses-tmseeg suffix for aalto
        ses_tag = ''            # no session in fmriprep filenames for aalto

        mri_anat = os.path.join(
            bids_root, bids_name, 'ses-mri01', 'anat',
            f'{bids_name}_ses-mri01_acq-basic_T1w.nii.gz'
        )
        fmri_func_dir = os.path.join(deriv, 'fmriprep', bids_name, 'func')

        coreg_dir = None
        coreg_matrix = None
        coreg_image = None

        tms_session = os.path.join(bids_root, 'sourcedata', 'TMS', f'session_{bids_id}.nbe')
        tms_raw = []

        eeg_rs = None
        eeg_emg = None

        eeg_prepro_dir = os.path.join(
            deriv, 'eegprepro-rest-rg', bids_name, 'ses-tmseeg', 'eeg'
        )
        eeg_clean = os.path.join(
            eeg_prepro_dir,
            f'{bids_name}_ses-tmseeg_task-rest_epo.fif'
        )
        mep_mat = os.path.join(deriv, 'emg', bids_name, 'AmpsM.mat')

    else:
        raise ValueError(f"Unknown site: {site!r}. Expected 'chieti' or 'aalto'.")

    # ------------------------------------------------------------------
    # Common paths (built from site-specific variables above)
    # ------------------------------------------------------------------
    eeg_raw = os.path.join(
        bids_root, bids_name, 'ses-tmseeg', 'eeg',
        f'{bids_name}_ses-tmseeg_task-rest_run-01_eeg.set'
    )

    fmri_func = os.path.join(
        fmri_func_dir,
        f'{bids_name}{ses_tag}_task-rest_space-{space}_desc-preproc_bold.nii.gz'
    )
    fmri_confounds = os.path.join(
        fmri_func_dir,
        f'{bids_name}{ses_tag}_task-rest_desc-confounds_timeseries.tsv'
    )
    fmri_mask = os.path.join(
        fmri_func_dir,
        f'{bids_name}{ses_tag}_task-rest_space-{space}_desc-brain_mask.nii.gz'
    )

    fs_dir = os.path.join(deriv, 'freesurfer', fs_subject)
    fs_src = os.path.join(fs_dir, f'{fs_subject}-oct6-src.fif')
    fs_trans = os.path.join(fs_dir, f'{fs_subject}_trans.fif')

    source_dir = os.path.join(deriv, 'source', bids_name)
    connectivity_dir = os.path.join(deriv, 'connectivity', bids_name)

    montage_dir = os.path.join(deriv, 'montage', bids_name)
    montage_fname = os.path.join(
        montage_dir, f'{bids_name}_ses-tmseeg_task-rest_desc-montage_dig.fif'
    )

    return SubjectFiles(
        # Identifiers
        site=site,
        bids_name=bids_name,
        bids_id=bids_id,
        fs_subject=fs_subject,
        eeg_id=eeg_id,
        tms_id=tms_id,
        bids_root=bids_root,
        deriv=deriv,
        # EEG
        eeg_raw=eeg_raw,
        eeg_clean=eeg_clean,
        eeg_prepro_dir=eeg_prepro_dir,
        eeg_rs=eeg_rs,
        eeg_emg=eeg_emg,
        # MEP
        mep_mat=mep_mat,
        # TMS
        tms_session=tms_session,
        tms_raw=tms_raw,
        # MRI
        mri_anat=mri_anat,
        # fMRI
        fmri_func=fmri_func,
        fmri_confounds=fmri_confounds,
        fmri_mask=fmri_mask,
        fmri_func_dir=fmri_func_dir,
        # Coregistration
        coreg_matrix=coreg_matrix,
        coreg_image=coreg_image,
        coreg_dir=coreg_dir,
        # FreeSurfer
        fs_dir=fs_dir,
        fs_src=fs_src,
        fs_trans=fs_trans,
        # Source / connectivity
        source_dir=source_dir,
        connectivity_dir=connectivity_dir,
        # Montage
        montage_dir=montage_dir,
        montage_fname=montage_fname,
    )
