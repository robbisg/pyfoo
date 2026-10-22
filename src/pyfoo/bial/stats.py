
import numpy as np
import pandas as pd
import pingouin
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.optimize import curve_fit
from scipy.stats import permutation_test as scipy_permutation_test


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bonferroni_label(p, n_tests):
    if p < 0.05 / n_tests:
        return '3'
    elif p < 0.01:
        return '2'
    elif p < 0.05:
        return '1'
    return '0'


# ── Curve models ──────────────────────────────────────────────────────────────

def linear(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def exponential(x, a, b):
    return a * np.exp(b * x)


# ── Core statistical primitives ───────────────────────────────────────────────

def ttest_two_groups(x_a, x_b, equal_var=False):
    """Independent t-test. Returns (t, p, dof)."""
    result = ttest_ind(x_a, x_b, equal_var=equal_var)
    dof = getattr(result, 'df', np.nan)
    return result.statistic, result.pvalue, dof


def pairwise_correlation(df, value_col, expertise_col='expertise'):
    """Pearson correlation between expertise and value_col using pingouin."""
    if len(df) < 3:
        return pd.DataFrame({'r': [np.nan], 'p-unc': [np.nan]})
    return pingouin.pairwise_corr(df, columns=[expertise_col, value_col])


def polynomial_fit_with_permutation(x, y, order, n_permutations=100):
    """
    Fit a polynomial of given order to (x, y) and test via permutation.
    Returns (r2, r2_threshold_95pct, significant_label).
    """
    from scipy.stats import pearsonr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    w = np.polyfit(x, y, order)
    y_pred = np.polyval(w, x)
    r2 = pearsonr(y, y_pred)[0]

    perm_r2 = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        w_perm = np.polyfit(x, y_perm, order)
        perm_r2.append(pearsonr(y_perm, np.polyval(w_perm, x))[0])

    threshold = np.percentile(perm_r2, 95)
    return r2, threshold, ('*' if r2 >= threshold else '')


def fit_function_with_permutation(df, x_col, y_col, func, n_permutations=1000):
    """
    Fit func to (x_col, y_col) and test significance with a permutation test.
    Returns (r_squared, pvalue, params).
    """
    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)

    params, _ = curve_fit(func, x, y)
    residuals = y - func(x, *params)
    r_squared = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)

    def _stat(x_s, y_s):
        try:
            p, _ = curve_fit(func, x_s, y_s)
            res = y_s - func(x_s, *p)
            return 1 - np.sum(res**2) / np.sum((y_s - np.mean(y_s))**2)
        except Exception:
            return 0.0

    perm = scipy_permutation_test(
        (x, y), _stat, n_resamples=n_permutations, alternative='two-sided'
    )
    return r_squared, perm.pvalue, params


# ── Loop-based analysis runners ───────────────────────────────────────────────

def run_pairwise_tests_by_group(df, value_col, group_col='group',
                                 condition_col='condition', padjust='fdr_bh'):
    """
    For each group, run pairwise tests between conditions with FDR correction.

    Mirrors::

        for group in df[group_col].unique():
            pingouin.pairwise_tests(df[group], dv=value_col,
                                    between=[condition_col], padjust=padjust)
    """
    results = []
    for group in df[group_col].unique():
        r = pingouin.pairwise_tests(
            df[df[group_col] == group],
            dv=value_col,
            between=[condition_col],
            padjust=padjust,
        )
        r[group_col] = group
        results.append(r)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def run_ttest_experts_vs_novices(df, value_col, network_col='network',
                                  condition_col='condition', group_col='group',
                                  expert_label='E', novice_label='N', n_bonferroni=7):
    """
    T-test comparing experts vs novices for every (network, condition) pair.

    Mirrors::

        for network in df[network_col].unique():
            for condition in df[condition_col].unique():
                ttest_ind(experts, novices)
    """
    results = []
    for network in df[network_col].unique():
        for condition in df[condition_col].unique():
            mask = np.logical_and(
                df[network_col] == network,
                df[condition_col] == condition,
            )
            x_e = df[np.logical_and(mask, df[group_col] == expert_label)][value_col].astype(float)
            x_n = df[np.logical_and(mask, df[group_col] == novice_label)][value_col].astype(float)
            if x_e.empty or x_n.empty:
                continue
            t, p, dof = ttest_two_groups(x_e, x_n)
            results.append({
                network_col: network, condition_col: condition,
                'mean_a': x_e.mean(), 'mean_b': x_n.mean(),
                'stdev_a': x_e.std(), 'stdev_b': x_n.std(),
                'test': f'{expert_label}-{novice_label}',
                't': t, 'p': p, 'df': dof,
                'significant': _bonferroni_label(p, n_bonferroni),
            })
    return pd.DataFrame(results)


def run_ttest_meditation_styles_group(df, value_col, network_col='network', group_col='group',
                                 condition_col='condition',
                                 style_a='Samatha', style_b='Vipassana', n_bonferroni=7):
    """
    T-test comparing FA (Samatha) vs OM (Vipassana) for every (network, group) pair.

    Mirrors::

        for network in df[network_col].unique():
            for group in df[group_col].unique():
                ttest_ind(samatha, vipassana)
    """
    results = []
    for network in df[network_col].unique():
        for group in df[group_col].unique():
            mask = np.logical_and(df[network_col] == network, df[group_col] == group)
            x_a = df[np.logical_and(mask, df[condition_col] == style_a)][value_col].astype(float)
            x_b = df[np.logical_and(mask, df[condition_col] == style_b)][value_col].astype(float)
            if x_a.empty or x_b.empty:
                continue
            t, p, dof = ttest_two_groups(x_a, x_b)
            results.append({
                network_col: network, group_col: group,
                'mean_a': x_a.mean(), 'mean_b': x_b.mean(),
                'stdev_a': x_a.std(), 'stdev_b': x_b.std(),
                'test': f'{style_a}-{style_b}',
                't': t, 'p': p, 'df': dof,
                'significant': _bonferroni_label(p, n_bonferroni),
            })
    return pd.DataFrame(results)


def run_ttest_meditation_styles(df, value_col, network_col='network', group_col='group',
                                 condition_col='condition',
                                 style_a='Samatha', style_b='Vipassana', n_bonferroni=7):
    """
    T-test comparing FA (Samatha) vs OM (Vipassana) for every (network, group) pair.

    Mirrors::

        for network in df[network_col].unique():
            for group in df[group_col].unique():
                ttest_ind(samatha, vipassana)
    """
    results = []
    for network in df[network_col].unique():
        mask = df[network_col] == network
        x_a = df[np.logical_and(mask, df[condition_col] == style_a)][value_col].astype(float)
        x_b = df[np.logical_and(mask, df[condition_col] == style_b)][value_col].astype(float)
        if x_a.empty or x_b.empty:
            continue
        t, p, dof = ttest_two_groups(x_a, x_b)
        results.append({
            network_col: network, 
            'mean_a': x_a.mean(), 'mean_b': x_b.mean(),
            'stdev_a': x_a.std(), 'stdev_b': x_b.std(),
            'test': f'{style_a}-{style_b}',
            't': t, 'p': p, 'df': dof,
            'significant': _bonferroni_label(p, n_bonferroni),
        })
    return pd.DataFrame(results)



def run_correlation_network_condition(df, value_col, expertise_col='expertise',
                                       network_col='network', condition_col='condition',
                                       group_col='group', expert_label='E'):
    """
    Pearson correlation between expertise and value_col in experts,
    for every (network, condition) pair.

    Mirrors::

        for network in df[network_col].unique():
            for condition in df[condition_col].unique():
                pingouin.pairwise_corr(experts, [expertise, value_col])
    """
    results = []
    for network in df[network_col].unique():
        for condition in df[condition_col].unique():
            mask = np.logical_and(
                df[network_col] == network,
                np.logical_and(df[condition_col] == condition, df[group_col] == expert_label),
            )
            df_ = df[mask]
            if df_.empty:
                continue
            corr = pairwise_correlation(df_, value_col, expertise_col)
            corr[network_col] = network
            corr[condition_col] = condition
            results.append(corr)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def run_polynomial_regression(df, value_col, expertise_col='expertise',
                               network_col='network', condition_col='condition',
                               group_col='group', expert_label='E',
                               orders=(1, 2), n_permutations=100):
    """
    Polynomial regression (expertise → value_col) with permutation significance test,
    for every (network, condition, order) combination in experts.

    Mirrors::

        for network in df[network_col].unique():
            for condition in df[condition_col].unique():
                for order in orders:
                    polyfit(x, y, order) + permutation_test
    """
    results = []
    for network in df[network_col].unique():
        for condition in df[condition_col].unique():
            mask = np.logical_and(
                df[network_col] == network,
                np.logical_and(df[condition_col] == condition, df[group_col] == expert_label),
            )
            df_ = df[mask]
            if df_.empty:
                continue
            x = df_[expertise_col].values
            y = df_[value_col].values
            for order in orders:
                r2, threshold, sig = polynomial_fit_with_permutation(x, y, order, n_permutations)
                results.append({
                    network_col: network, condition_col: condition,
                    'order': order, 'r2': r2, 'r2_threshold': threshold,
                    'significant': sig,
                })
    return pd.DataFrame(results)


def run_ttest_energy_transitions(df, value_col, network_col='network',
                                  start_col='start_condition', end_col='end_condition',
                                  group_col='group', start_conditions=None,
                                  expert_label='E', novice_label='N', n_bonferroni=7):
    """
    T-test comparing experts vs novices for every (network, start, end) transition.

    Mirrors::

        for network in df[network_col].unique():
            for start_condition in start_conditions:
                for end_condition in df[end_col].unique():
                    ttest_ind(experts, novices)
    """
    if start_conditions is None:
        start_conditions = df[start_col].unique()

    results = []
    for network in df[network_col].unique():
        for start in start_conditions:
            for end in df[end_col].unique():
                mask = np.logical_and(df[network_col] == network, df[start_col] == start)
                mask = np.logical_and(mask, df[end_col] == end)
                x_e = df[np.logical_and(mask, df[group_col] == expert_label)][value_col].astype(float)
                x_n = df[np.logical_and(mask, df[group_col] == novice_label)][value_col].astype(float)
                if x_e.empty or x_n.empty:
                    continue
                t, p, dof = ttest_two_groups(x_e, x_n)
                results.append({
                    network_col: network, start_col: start, end_col: end,
                    'test': f'{expert_label}-{novice_label}',
                    't': t, 'p': p, 'df': dof,
                    'significant': _bonferroni_label(p, n_bonferroni),
                })
    return pd.DataFrame(results)


def run_function_fit_expertise(df, value_col, expertise_col='expertise',
                                network_col='network', end_col='end_condition',
                                start_col='start_condition', start_value='Rest',
                                group_col='group', expert_label='E',
                                funcs=None, n_permutations=1000):
    """
    Fit linear and quadratic functions to expertise vs value_col in experts,
    for every (network, end_condition) pair, starting from start_value.

    Mirrors::

        for network in df[network_col].unique():
            for condition in df[end_col].unique():
                fit_and_plot(experts, expertise, value_col, linear)
                fit_and_plot(experts, expertise, value_col, quadratic)
    """
    if funcs is None:
        funcs = {'linear': linear, 'quadratic': quadratic}

    results = []
    for network in df[network_col].unique():
        for condition in df[end_col].unique():
            mask = np.logical_and(df[network_col] == network, df[end_col] == condition)
            if start_col is not None:
                mask = np.logical_and(mask, df[start_col] == start_value)
            mask = np.logical_and(mask, df[group_col] == expert_label)
            df_sub = df[mask]
            if df_sub.empty:
                continue
            for fname, func in funcs.items():
                try:
                    r2, pval, _ = fit_function_with_permutation(
                        df_sub, expertise_col, value_col, func, n_permutations
                    )
                    results.append({
                        network_col: network, end_col: condition,
                        'func': fname, 'r2': r2, 'pvalue': pval,
                        'significant': '*' if pval < 0.05 else '',
                    })
                except Exception:
                    pass
    return pd.DataFrame(results)


def run_node_level_analysis(df, value_col, conditions=None, node_col='node',
                             condition_col='condition', start_col=None, end_col=None,
                             start_value='Rest', group_col='group',
                             expertise_col='expertise',
                             expert_label='E', novice_label='N', n_bonferroni=79):
    """
    Node-level t-test (experts vs novices) and expertise correlation,
    for every (condition, node) pair.  Returns arrays ready for surface plotting.

    Mirrors::

        for condition in conditions:
            for node in df[node_col].unique():
                ttest_ind(experts, novices)
                pingouin.pairwise_corr(experts, [expertise, value_col])
    """
    if conditions is None:
        conditions = df[condition_col].unique() if end_col is None else df[end_col].unique()

    surface_results = {}
    all_rows = []

    for condition in conditions:
        if start_col is not None:
            mask_cond = np.logical_and(df[start_col] == start_value, df[end_col] == condition)
        else:
            mask_cond = df[condition_col] == condition

        t_atlas, p_t_atlas, r_atlas, p_r_atlas = [], [], [], []

        for node in df[node_col].unique():
            mask_node = np.logical_and(df[node_col] == node, mask_cond)
            x_e = df[np.logical_and(mask_node, df[group_col] == expert_label)][value_col].astype(float)
            x_n = df[np.logical_and(mask_node, df[group_col] == novice_label)][value_col].astype(float)

            t, p = np.nan, np.nan
            if not x_e.empty and not x_n.empty:
                res = ttest_ind(x_e, x_n)
                t, p = res.statistic, res.pvalue

            corr = pairwise_correlation(df[np.logical_and(mask_node, df[group_col] == expert_label)],
                                        value_col, expertise_col)
            r = corr['r'].values[0]
            p_r = corr['p-unc'].values[0]

            t_atlas.append(t)
            p_t_atlas.append(p)
            r_atlas.append(r)
            p_r_atlas.append(p_r)

            all_rows.append({
                node_col: node, condition_col: condition,
                't': t, 'p_t': p, 'r': r, 'p_r': p_r,
                'mean_e': x_e.mean(), 'mean_n': x_n.mean(),
                'stdev_e': x_e.std(), 'stdev_n': x_n.std(),
                'significant_t': _bonferroni_label(p, n_bonferroni) if not np.isnan(p) else '0',
                'significant_r': _bonferroni_label(p_r, n_bonferroni) if not np.isnan(p_r) else '0',
            })

        surface_results[condition] = {
            't_atlas': np.array(t_atlas),
            'p_t_atlas': np.array(p_t_atlas),
            'r_atlas': np.array(r_atlas),
            'p_r_atlas': np.array(p_r_atlas),
        }

    return surface_results, pd.DataFrame(all_rows)


def run_stability_condition_tests(df, value_col, condition_col='condition',
                                   group_col='group', expert_label='E', novice_label='N'):
    """
    T-test for every condition (experts vs novices) and for every group (FA vs OM).

    Mirrors::

        for condition in df[condition_col].unique():
            ttest_ind(experts, novices)

        for group in df[group_col].unique():
            ttest_ind(samatha, vipassana)
    """
    results = []

    for condition in df[condition_col].unique():
        mask = df[condition_col] == condition
        x_e = df[np.logical_and(mask, df[group_col] == expert_label)][value_col].astype(float)
        x_n = df[np.logical_and(mask, df[group_col] == novice_label)][value_col].astype(float)
        t, p, dof = ttest_two_groups(x_e, x_n)
        results.append({condition_col: condition, 'test': f'{expert_label}-{novice_label}',
                         't': t, 'p': p, 'df': dof})

    for group in df[group_col].unique():
        mask = df[group_col] == group
        x_a = df[np.logical_and(mask, df[condition_col] == 'Samatha')][value_col].astype(float)
        x_b = df[np.logical_and(mask, df[condition_col] == 'Vipassana')][value_col].astype(float)
        if x_a.empty or x_b.empty:
            continue
        t, p, dof = ttest_two_groups(x_a, x_b)
        results.append({group_col: group, 'test': 'Samatha-Vipassana',
                         't': t, 'p': p, 'df': dof})

    return pd.DataFrame(results)


def run_landscape_complexity_tests(df, value_col, condition_col='condition',
                                    group_col='group', expert_label='E', novice_label='N'):
    """
    T-test and Mann-Whitney U test for every condition (experts vs novices)
    and for every group (FA vs OM).

    Mirrors::

        for condition in df[condition_col].unique():
            ttest_ind(experts, novices)
            mannwhitneyu(experts, novices)

        for group in df[group_col].unique():
            ttest_ind(samatha, vipassana)
            mannwhitneyu(samatha, vipassana)
    """
    results = []

    for condition in df[condition_col].unique():
        mask = df[condition_col] == condition
        x_e = df[np.logical_and(mask, df[group_col] == expert_label)][value_col].astype(float)
        x_n = df[np.logical_and(mask, df[group_col] == novice_label)][value_col].astype(float)
        t, p_t, dof = ttest_two_groups(x_e, x_n)
        U, p_u = mannwhitneyu(x_e, x_n)
        results.append({
            condition_col: condition, 'test': f'{expert_label}-{novice_label}',
            't': t, 'p_t': p_t, 'df': dof, 'U': U, 'p_u': p_u,
        })

    for group in df[group_col].unique():
        mask = df[group_col] == group
        x_a = df[np.logical_and(mask, df[condition_col] == 'Samatha')][value_col].astype(float)
        x_b = df[np.logical_and(mask, df[condition_col] == 'Vipassana')][value_col].astype(float)
        if x_a.empty or x_b.empty:
            continue
        t, p_t, dof = ttest_two_groups(x_a, x_b)
        U, p_u = mannwhitneyu(x_a, x_b)
        results.append({
            group_col: group, 'test': 'Samatha-Vipassana',
            't': t, 'p_t': p_t, 'df': dof, 'U': U, 'p_u': p_u,
        })

    return pd.DataFrame(results)


def run_controllability_analysis(df, keys=None, network_col='network',
                                  condition_col='condition', group_col='group',
                                  expert_label='E', novice_label='N',
                                  expertise_col='expertise',
                                  n_bonferroni=7, n_permutations=100):
    """
    Run all controllability analyses for each metric key.

    Mirrors the outer selector loop::

        for key in keys:  # ['average_controllability', 'modal_controllability']
            run_ttest_experts_vs_novices(df, key, ...)
            run_ttest_meditation_styles(df, key, ...)
            run_correlation_network_condition(df, key, ...)
            run_polynomial_regression(df, key, ...)
    """
    if keys is None:
        keys = ['average_controllability', 'modal_controllability']

    all_results = {}
    for key in keys:
        all_results[key] = {
            'ttest_experts_novices': run_ttest_experts_vs_novices(
                df, key, network_col=network_col, condition_col=condition_col,
                group_col=group_col, expert_label=expert_label,
                novice_label=novice_label, n_bonferroni=n_bonferroni,
            ),
            'ttest_meditation_styles': run_ttest_meditation_styles(
                df, key, network_col=network_col, group_col=group_col,
                condition_col=condition_col, n_bonferroni=n_bonferroni,
            ),
            'correlation_expertise': run_correlation_network_condition(
                df, key, expertise_col=expertise_col, network_col=network_col,
                condition_col=condition_col, group_col=group_col,
                expert_label=expert_label,
            ),
            'polynomial_regression': run_polynomial_regression(
                df, key, expertise_col=expertise_col, network_col=network_col,
                condition_col=condition_col, group_col=group_col,
                expert_label=expert_label, n_permutations=n_permutations,
            ),
        }
    return all_results


# ── Surface plotting utilities ────────────────────────────────────────────────

def network_values_to_node_vector(network_values, n_rois=300, resolution_mm=2,
                                   fill_value=0.0, labels=None):
    """
    Expand per-network scalar values to a per-node vector for plot_surface_schaefer.

    Each of the n_rois nodes is assigned the value of the network it belongs to,
    as determined by the Schaefer atlas label (3rd '_'-split field, e.g. 'Vis').
    Nodes whose network is not in network_values are set to fill_value.

    Parameters
    ----------
    network_values : dict or pd.Series
        Mapping from network name to scalar value, e.g.::

            {'Vis': 0.8, 'SomMot': -0.3, 'DorsAttn': 0.5}

        Typical sources:

        - ``results_df.set_index('network')['t']``  (one condition slice)
        - ``ttest_df[ttest_df['condition'] == 'Rest'].set_index('network')['t']``

    n_rois : int
        Number of parcels; must match what was passed to plot_surface_schaefer.
    resolution_mm : int
        Schaefer atlas resolution in mm.
    fill_value : float
        Value assigned to nodes whose network has no entry in network_values.
    labels : list of str, optional
        Pre-loaded Schaefer label list (skips the nilearn fetch when provided).

    Returns
    -------
    np.ndarray of shape (n_rois,)
        Ready for ``plot_surface_schaefer(node_vector)``.

    Examples
    --------
    >>> ttest_rest = ttest_en[ttest_en['condition'] == 'Rest'].set_index('network')['t']
    >>> node_vec = network_values_to_node_vector(ttest_rest)
    >>> plot_surface_schaefer(node_vec, cmap='RdBu', title='T (E vs N, Rest)')
    """
    from nilearn.datasets import fetch_atlas_schaefer_2018

    if labels is None:
        raw = fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=resolution_mm)['labels']
        labels = [n.decode('utf-8') for n in raw]

    networks = [l.split('_')[2] for l in labels]

    node_vector = np.full(n_rois, fill_value, dtype=float)
    for i, net in enumerate(networks):
        if net in network_values:
            node_vector[i] = network_values[net]

    return node_vector


def results_to_node_vector(results_df, value_col, condition=None,
                            condition_col='condition', network_col='network',
                            n_rois=300, resolution_mm=2, fill_value=0.0, labels=None):
    """
    Slice a results DataFrame for one condition, then call network_values_to_node_vector.

    Convenience wrapper so common plotting patterns collapse to two lines::

        node_vec = results_to_node_vector(ttest_en, 't', condition='Rest')
        plot_surface_schaefer(node_vec, cmap='RdBu', title='T (E vs N, Rest)')

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_ttest_experts_vs_novices, run_ttest_meditation_styles, etc.
    value_col : str
        Column to use as the scalar value (e.g. 't', 'r', 'p').
    condition : str or None
        If given, filter rows where condition_col == condition first.
        If None, uses all rows (assumes df is already filtered to one condition).
    condition_col, network_col : str
        Column names for condition and network.
    n_rois, resolution_mm, fill_value, labels :
        Passed through to network_values_to_node_vector.

    Returns
    -------
    np.ndarray of shape (n_rois,)
    """
    df = results_df
    if condition is not None:
        df = results_df[results_df[condition_col] == condition]

    mapping = df.set_index(network_col)[value_col]
    return network_values_to_node_vector(mapping, n_rois=n_rois,
                                          resolution_mm=resolution_mm,
                                          fill_value=fill_value, labels=labels)
