# Return montage from NexstimSession and reordering files.
import os

from mne.channels import read_custom_montage
import mne
import numpy as np

import os
import os.path as op
from pathlib import Path
import re
import shutil as sh

import mne
from mne.utils import logger
import numpy as np
from scipy.io import loadmat, savemat
from mne_bids.path import _parse_ext


CAPTIONS = {
    "00_sensors":
        "Every electrode should sit in a plausible 10-20/10-10 location. A "
        "channel at the wrong spot means montage/channel-name mismatch - fix "
        "before referencing, interpolation or ICA.",
    "01_raw_timeseries":
        "Traces within ~+/-50-100 uV on a flat baseline. Scan for flat lines "
        "(dead electrodes), single noisy channels, slow drift, frontal blinks "
        "and high-frequency muscle fuzz.",
    "01_raw_psd":
        "Expect a 1/f downslope plus a posterior alpha bump (8-12 Hz). The "
        "sharp 50/60 Hz spike is the line noise removed next - confirm it is "
        "present now so you can confirm it is gone later.",
    "03_psd_after_notch":
        "The mains spike and harmonics should be gone; the 1/f slope and alpha "
        "peak unchanged (a narrow notch, not a wide gouge). Power below the "
        "high-pass should roll off.",
    "04_bad_channels":
        "Channels flagged bad (red) should truly look flat/noisy/de-correlated. "
        "Many flagged (>10-15%) suggests a global problem, not bad electrodes. "
        "Un-flag any good channel before ICA.",
    "06_ica_components":
        "Recognise artifact maps: blink = frontal hotspot; saccade = "
        "left-right dipole; heart = broad weak gradient; muscle = edge-focal, "
        "high-freq; line = focal with 50/60 Hz peak. Keep dipolar brain comps.",
    "06_ica_flagged_properties":
        "For each component being removed, confirm topomap + spectrum + time "
        "course all agree it is an artifact. If one looks like brain (dipolar "
        "map, 1/f+peak spectrum), take it out of the exclude list.",
    "06_ica_overlay":
        "Red = before, black = after removal. Frontal blink deflections should "
        "shrink, but posterior alpha / background should be almost unchanged. "
        "If everything changes, you are removing brain signal.",
    "09_autoreject_log":
        "Epochs x channels grid. Mostly blank = good. Bad rows = noisy time "
        "segments (a few dropped is fine). Bad columns = a channel missed at "
        "step 4. Huge rejection = revisit thresholds/earlier steps.",
    "09_drop_log":
        "How many epochs survived. Losing a small fraction is normal; losing "
        "the majority means something upstream is wrong.",
    "10_clean_psd":
        "Final check: 1/f slope intact, clean ~10 Hz alpha peak, NO 50/60 Hz "
        "spike, no broadband muscle plateau. Signature of analysis-ready EEG.",
    "10_alpha_topomap":
        "KEY resting-state sanity check: alpha (8-12 Hz) should be MAXIMAL over "
        "posterior (occipital/parietal) sites. Frontal-max or single-electrode "
        "hotspots indicate a reference problem or residual bad channel.",
}

def interpolate_missing_channels_position(epochs, montage):
    # Define the standard montage and adjacency for interpolation
    BVEF_PATH = '/home/robbis/git/BC-TMS-128.bvef'    
    
    standard_montage = read_custom_montage(BVEF_PATH)
    channel_names = standard_montage.ch_names[2:]  # Exclude non-EEG channels
    
    standard_info = mne.create_info(ch_names=channel_names, sfreq=1000.0, ch_types='eeg')
    standard_info.set_montage(standard_montage)
    adjacency, adj_chnames = mne.channels.find_ch_adjacency(standard_info, ch_type='eeg')
        
    # Identify missing channels
    missing_channels = [ch for ch in epochs.ch_names if ch not in montage.ch_names]
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


def savefig(report, fig, name, figdir=None):
    """Save a figure (or list of figures) to FIG_DIR and add it to the report."""
    figs = fig if isinstance(fig, (list, tuple)) else [fig]
    figs = [f for f in figs if f is not None]
    if not figs:
        return
    # save PNGs
    for i, f in enumerate(figs):
        try:
            suffix = "" if len(figs) == 1 else f"_{i}"
            f.savefig(os.path.join(figdir, f"{name}{suffix}.png"),
                      dpi=130, bbox_inches="tight")
        except Exception:
            pass
    # add to the HTML report, using the guidance text as caption
    raw_caption = CAPTIONS.get(name, "")
    caption = f"WHAT TO LOOK AT\n\n{raw_caption}" if raw_caption else ""
    section = name.split("_", 1)[0]
    try:
        if len(figs) == 1:
            report.add_figure(figs[0], title=name, caption=caption,
                              section=section, replace=True)
        else:
            titles = [f"{name} [{i}]" for i in range(len(figs))]
            captions = [caption] * len(figs)
            report.add_figure(figs, title=titles, caption=captions,
                              section=section, replace=True)
    except Exception as err:
        print(f"  (could not add {name} to report: {err})")
        


def copyfile_eeglab(src, dest):
    """Copy an EEGLAB file to a new location.

    If the EEGLAB ``.set`` file comes with an accompanying ``.fdt`` binary file
    that contains the actual data, this function will copy this file, too, and
    update all internal pointers in the new ``.set`` file.

    Parameters
    ----------
    src : path-like
        Path to the source raw .set file.
    dest : path-like
        Path to the destination of the new .set file.

    See Also
    --------
    copyfile_brainvision
    copyfile_bti
    copyfile_ctf
    copyfile_edf
    copyfile_kit

    """
    # Get extension of the EEGLAB file
    _, ext_src = _parse_ext(src)
    fname_dest, ext_dest = _parse_ext(dest)
    if ext_src != ext_dest:
        raise ValueError(
            f"Need to move data with same extension" f" but got {ext_src}, {ext_dest}"
        )

    # Load the EEG struct
    # NOTE: do *not* simplify cells, because this changes the underlying
    # structure and potentially breaks re-reading of the file
    uint16_codec = None
    eeg = loadmat(
        file_name=src,
        simplify_cells=False,
        appendmat=False,
        uint16_codec=uint16_codec,
        mat_dtype=True,
        squeeze_me=True
    )
    oldstyle = False
    if "EEG" in eeg:
        eeg = eeg["EEG"]
        oldstyle = True

    has_fdt_link = False
    try:
        # If the data field is a string, it points to a .fdt file in src dir
        if isinstance(eeg["data"][0], str):
            has_fdt_link = True
            fdt_fname = eeg["data"]
            
        if isinstance(eeg["data"][0], np.ndarray) and eeg["data"][0].dtype.kind == "U":
            has_fdt_link = True
            fdt_fname = eeg["data"][0][0]
            
    except KeyError:
        pass
    except TypeError:
        pass
    except IndexError:
        pass

    if has_fdt_link:
        #fdt_fname = eeg["data"][0]

        assert fdt_fname.endswith(".fdt"), f"Unexpected fdt name: {fdt_fname}"
        head, _ = op.split(src)
        fdt_path = op.join(head, fdt_fname)
        
        # Copy the .fdt file and give it a new name
        fdt_name_new = fname_dest + ".fdt"
        
        if fdt_path == fdt_name_new:
            logger.warning(f"Source and destination .fdt files are the same: {fdt_path}. Skipping copy.")        
            return
        
        if op.exists(fdt_path):
            sh.copyfile(fdt_path, fdt_name_new)

        # Now adjust the pointer in the .set file
        # NOTE: Clunky numpy code is to match MATLAB structure for "savemat"
        _, tail = op.split(fdt_name_new)
        new_value = np.empty((1, 1), dtype=object)
        new_value[0, 0] = np.atleast_1d(np.array(tail))
        eeg["data"] = new_value

        # Save the EEG dictionary as a Matlab struct again
        mdict = dict(EEG=eeg) if oldstyle else eeg
        savemat(file_name=dest, mdict=mdict, appendmat=False)
        logger.info(f"Wrote {dest} ")
    else:
        # If no .fdt file, simply copy the .set file, no modifications
        # necessary
        sh.copyfile(src, dest)