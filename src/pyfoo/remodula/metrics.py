import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from sklearn.metrics import recall_score, make_scorer


def get_classification_scoring():
    """Scoring dict for `sklearn.model_selection.cross_validate` that adds
    AUC, sensitivity, specificity and a calibration index (Brier score)
    to the usual balanced accuracy.
    """
    return {
        'balanced_accuracy': 'balanced_accuracy',
        'auc': 'roc_auc',
        'sensitivity': make_scorer(recall_score, pos_label=True),
        'specificity': make_scorer(recall_score, pos_label=False),
        'brier': 'neg_brier_score',
    }


def summarize_metrics(repeat_scores, ci=0.95):
    """Summarize repeated cross-validation scores with a percentile
    confidence interval computed across repeats.

    Parameters
    ----------
    repeat_scores : dict[str, array-like]
        One array per metric, each of shape (n_repeats,). A 'brier' entry
        is expected to hold `neg_brier_score` values and is flipped back
        to the Brier score (lower = better calibrated).
    ci : float
        Confidence level, e.g. 0.95 for a 95% CI.

    Returns
    -------
    pd.DataFrame with columns ['metric', 'mean', 'std', 'ci_lower', 'ci_upper']
    """
    alpha = (1 - ci) / 2
    rows = []
    for metric, values in repeat_scores.items():
        values = np.asarray(values, dtype=float)
        if metric == 'brier':
            values = -values

        lo, hi = np.percentile(values, [100 * alpha, 100 * (1 - alpha)])
        rows.append({
            'metric': metric,
            'mean': values.mean(),
            'std': values.std(),
            'ci_lower': lo,
            'ci_upper': hi,
        })

    return pd.DataFrame(rows)


def permutation_pvalue(observed, null_distribution, higher_is_better=True):
    """One-sided permutation p-value with the Phipson & Smyth (+1/+1)
    correction, which avoids a p-value of exactly 0.

    `null_distribution` is expected on the same scale/convention as
    `observed` (e.g. both `neg_brier_score`, where higher is better).
    """
    null_distribution = np.asarray(null_distribution, dtype=float)
    n = null_distribution.shape[0]
    if higher_is_better:
        n_as_extreme = np.count_nonzero(null_distribution >= observed)
    else:
        n_as_extreme = np.count_nonzero(null_distribution <= observed)

    return (n_as_extreme + 1) / (n + 1)


def summarize_permutation(observed, null_scores):
    """One-sided permutation p-values for a set of metrics.

    Parameters
    ----------
    observed : dict[str, float]
        Observed (non-permuted) mean score per metric.
    null_scores : dict[str, array-like]
        Null-distribution scores per metric (one value per permutation),
        sharing `observed`'s keys. Every metric follows scikit-learn's
        "higher is better" scoring convention, including 'brier', which
        here is the signed `neg_brier_score`.

    Returns
    -------
    pd.DataFrame with columns ['metric', 'observed', 'p_value']
    """
    rows = []
    for metric, null_values in null_scores.items():
        p = permutation_pvalue(observed[metric], null_values)
        rows.append({
            'metric': metric,
            'observed': observed[metric],
            'p_value': p,
        })

    return pd.DataFrame(rows)


def plot_metrics_ci(df_summary, color='teal', title=None, ax=None):
    """Point estimate + CI error-bar plot for a metrics summary table
    produced by `summarize_metrics`.
    """
    if ax is None:
        fig, ax = pl.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    xerr = np.vstack([
        df_summary['mean'] - df_summary['ci_lower'],
        df_summary['ci_upper'] - df_summary['mean'],
    ])

    ax.errorbar(df_summary['mean'], df_summary['metric'], xerr=xerr,
                fmt='o', color=color, capsize=4)
    ax.set_xlabel('Score')
    if title is not None:
        ax.set_title(title)

    pl.tight_layout()

    return fig, ax
