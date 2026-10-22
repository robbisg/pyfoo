import mne
import numpy as np
from pathlib import Path


class SourceSpaceTransformer:
    """Forward + inverse pipeline for EEG/MEG source reconstruction.

    Follows the scikit-learn fit/transform convention:
    - fit(epochs)         : builds forward solution, noise covariance, inverse operator
    - transform(epochs)   : applies the inverse and returns a list of STCs
    - fit_transform(epochs): fit + transform in one call

    The source space and BEM are computed lazily from fsaverage on the first
    call to fit() and reused for subsequent calls (same template, different
    subjects/sessions share the same geometry).

    Parameters
    ----------
    method : str
        Inverse method: 'eLORETA', 'MNE', or 'sLORETA'.
    snr : float
        Signal-to-noise ratio used to derive lambda2 = 1 / snr**2.
    subject_fs : str
        FreeSurfer subject name for the template (default 'fsaverage').
    subjects_dir : str or Path or None
        FreeSurfer subjects directory. If None, fetched automatically via
        mne.datasets.fetch_fsaverage().
    spacing : str
        Source space downsampling (e.g. 'oct6').
    ico : int
        Icosahedron order for BEM model tessellation.
    trans : str
        Head-to-MRI transform. Use 'fsaverage' for the built-in identity.
    mindist : float
        Minimum distance (mm) from sources to the inner skull surface.
    cov_tmax : float
        Upper time limit (s) used for noise covariance estimation.
    cov_method : str
        Covariance estimator passed to mne.compute_covariance().
    fixed : bool
        Whether to use fixed source orientations.
    loose : float or 'auto'
        Loose orientation constraint for the inverse operator.
    depth : float or None
        Depth weighting for the inverse operator (None = disabled).
    pick_ori : str or None
        Orientation selection when applying the inverse.
    """

    def __init__(
        self,
        method="eLORETA",
        snr=3.0,
        subject_fs="fsaverage",
        subjects_dir=None,
        spacing="oct6",
        ico=4,
        trans="fsaverage",
        mindist=5.0,
        cov_tmax=0.0,
        cov_method="empirical",
        fixed=True,
        loose="auto",
        depth=None,
        pick_ori=None,
    ):
        self.method = method
        self.snr = snr
        self.subject_fs = subject_fs
        self.subjects_dir = subjects_dir
        self.spacing = spacing
        self.ico = ico
        self.trans = trans
        self.mindist = mindist
        self.cov_tmax = cov_tmax
        self.cov_method = cov_method
        self.fixed = fixed
        self.loose = loose
        self.depth = depth
        self.pick_ori = pick_ori

        self._src = None
        self._bem = None
        self.forward_ = None
        self.noise_cov_ = None
        self.inverse_operator_ = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_source_space(self):
        if self._src is not None:
            return

        if self.subjects_dir is None:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
            self.subjects_dir = Path(fs_dir).parent

        self._src = mne.setup_source_space(
            subject=self.subject_fs,
            spacing=self.spacing,
            subjects_dir=self.subjects_dir,
            add_dist=False,
        )
        bem_model = mne.make_bem_model(
            subject=self.subject_fs,
            ico=self.ico,
            subjects_dir=self.subjects_dir,
        )
        self._bem = mne.make_bem_solution(bem_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def src(self):
        self._ensure_source_space()
        return self._src

    def fit(self, epochs):
        """Compute forward solution, noise covariance, and inverse operator.

        Parameters
        ----------
        epochs : mne.Epochs

        Returns
        -------
        self
        """
        self._ensure_source_space()

        fwd = mne.make_forward_solution(
            epochs.info,
            trans=self.trans,
            src=self._src,
            bem=self._bem,
            eeg=True,
            mindist=self.mindist,
        )
        self.forward_ = mne.convert_forward_solution(
            fwd, surf_ori=True, force_fixed=True
        )

        self.noise_cov_ = mne.compute_covariance(
            epochs,
            tmax=self.cov_tmax,
            method=self.cov_method,
            rank=None,
        )

        self.inverse_operator_ = mne.minimum_norm.make_inverse_operator(
            epochs.info,
            self.forward_,
            self.noise_cov_,
            fixed=self.fixed,
            loose=self.loose,
            depth=self.depth,
        )

        return self

    def transform(self, epochs):
        """Apply the inverse operator to epochs and return source estimates.

        Parameters
        ----------
        epochs : mne.Epochs

        Returns
        -------
        stcs : list of mne.SourceEstimate
        """
        if self.inverse_operator_ is None:
            raise RuntimeError("Call fit() before transform().")

        lambda2 = 1.0 / self.snr ** 2

        return mne.minimum_norm.apply_inverse_epochs(
            epochs,
            self.inverse_operator_,
            lambda2=lambda2,
            method=self.method,
            pick_ori=self.pick_ori,
            return_generator=False,
        )

    def fit_transform(self, epochs):
        """Fit on epochs and return source estimates.

        Parameters
        ----------
        epochs : mne.Epochs

        Returns
        -------
        stcs : list of mne.SourceEstimate
        """
        return self.fit(epochs).transform(epochs)


class LabelTimeCoursesTransformer:
    """Extract parcel-averaged time courses from source estimates.

    Wraps mne.extract_label_time_course so it fits into a transform chain.

    Parameters
    ----------
    labels : list of mne.Label
        Parcellation labels (e.g. from pyfoo.atlases.parcellation).
    src : mne.SourceSpaces
        Source space used for the inverse solution.
    mode : str
        Averaging mode passed to mne.extract_label_time_course
        (default 'mean_flip').
    """

    def __init__(self, labels, src, mode="mean_flip"):
        self.labels = labels
        self.src = src
        self.mode = mode

    def transform(self, stcs):
        """Extract label time courses from a list of STCs.

        Parameters
        ----------
        stcs : list of mne.SourceEstimate

        Returns
        -------
        label_ts : list of ndarray, shape (n_labels, n_times)
        """
        return mne.extract_label_time_course(
            stcs,
            self.labels,
            self.src,
            mode=self.mode,
            return_generator=False,
        )
