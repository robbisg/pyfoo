import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import gamma as gamma_func
from scipy.signal import find_peaks
import warnings
from joblib import Parallel, delayed

def extract_median(data, **kwargs):
    """Extract median peak-to-peak amplitude"""
    return np.median(data)

def extract_iqr(data, **kwargs):
    """Extract interquartile range"""
    return np.percentile(data, 75) - np.percentile(data, 25)

def extract_mad(data, **kwargs):
    """Extract median absolute deviation"""
    return np.median(np.abs(data - np.median(data)))

def extract_percentile(data, percentile=95, **kwargs):
    """Extract specified percentile"""
    return np.percentile(data, percentile)

def extract_cv(data, **kwargs):
    """Extract coefficient of variation (std/mean)"""
    mean_val = np.mean(data)
    if mean_val == 0:
        return np.nan
    return np.std(data) / mean_val

def extract_trimmed_mean(data, trim_percent=10, **kwargs):
    """Extract trimmed mean (removes specified percentage from both ends)"""
    return stats.trim_mean(data, trim_percent/100)

def extract_range(data, **kwargs):
    """Extract range (max - min)"""
    return np.max(data) - np.min(data)

def extract_quantile_skewness(data, **kwargs):
    """Extract quantile-based skewness: (Q3 + Q1 - 2*Q2) / (Q3 - Q1)"""
    q1, q2, q3 = np.percentile(data, [25, 50, 75])
    if q3 == q1:
        return 0
    return (q3 + q1 - 2*q2) / (q3 - q1)

def extract_l_moment_1(data, **kwargs):
    """Extract L-moment 1 (L-location, equivalent to mean)"""
    return np.mean(data)

def extract_l_moment_2(data, **kwargs):
    """Extract L-moment 2 (L-scale)"""
    sorted_data = np.sort(data)
    n = len(data)
    l2 = 0
    for i in range(n):
        l2 += (2*i - n + 1) * sorted_data[i]
    return l2 / (n * (n-1))

def extract_l_skewness(data, **kwargs):
    """Extract L-skewness (L3/L2)"""
    l2 = extract_l_moment_2(data)
    if l2 == 0:
        return 0
    
    sorted_data = np.sort(data)
    n = len(data)
    l3 = 0
    for i in range(n):
        l3 += ((i-1)*i - (n-i-2)*(n-i-1)) * sorted_data[i]
    l3 = l3 / (n * (n-1) * (n-2))
    
    return l3 / l2

def extract_proportion_above_threshold(data, threshold=None, **kwargs):
    """Extract proportion of values above threshold (default: median)"""
    if threshold is None:
        threshold = np.median(data)
    return np.mean(data > threshold)

def extract_gamma_shape(data, **kwargs):
    """Extract shape parameter from gamma distribution fit"""
    try:
        # Remove non-positive values for gamma fitting
        positive_data = data[data > 0]
        if len(positive_data) < 2:
            return np.nan
        
        shape, loc, scale = stats.gamma.fit(positive_data, floc=0)
        return shape
    except:
        return np.nan

def extract_gamma_scale(data, **kwargs):
    """Extract scale parameter from gamma distribution fit"""
    try:
        # Remove non-positive values for gamma fitting
        positive_data = data[data > 0]
        if len(positive_data) < 2:
            return np.nan
        
        shape, loc, scale = stats.gamma.fit(positive_data, floc=0)
        return scale
    except:
        return np.nan

def extract_lognormal_mu(data, **kwargs):
    """Extract mu parameter from log-normal distribution fit"""
    try:
        # Remove non-positive values for log-normal fitting
        positive_data = data[data > 0]
        if len(positive_data) < 2:
            return np.nan
        
        shape, loc, scale = stats.lognorm.fit(positive_data, floc=0)
        return np.log(scale)  # mu parameter
    except:
        return np.nan

def extract_lognormal_sigma(data, **kwargs):
    """Extract sigma parameter from log-normal distribution fit"""
    try:
        # Remove non-positive values for log-normal fitting
        positive_data = data[data > 0]
        if len(positive_data) < 2:
            return np.nan
        
        shape, loc, scale = stats.lognorm.fit(positive_data, floc=0)
        return shape  # sigma parameter
    except:
        return np.nan

def extract_normal_mean(data, **kwargs):
    """Extract mean from normal distribution fit"""
    return np.mean(data)

def extract_normal_std(data, **kwargs):
    """Extract standard deviation from normal distribution fit"""
    return np.std(data, ddof=1)

def extract_exponential_rate(data, **kwargs):
    """Extract rate parameter from exponential distribution fit"""
    try:
        # Shift data to ensure all values are positive
        min_val = np.min(data)
        shifted_data = data - min_val + 1e-8
        
        loc, scale = stats.expon.fit(shifted_data, floc=0)
        return 1/scale  # rate parameter (lambda)
    except:
        return np.nan

# ============================================================================
# ROBUST ESTIMATORS
# ============================================================================

def extract_huber_location(data, c=1.345, **kwargs):
    """Extract Huber M-estimator for location (robust mean)"""
    try:
        from scipy.optimize import minimize_scalar
        
        def huber_loss(mu):
            residuals = data - mu
            abs_residuals = np.abs(residuals)
            return np.sum(np.where(abs_residuals <= c, 
                                 0.5 * residuals**2,
                                 c * abs_residuals - 0.5 * c**2))
        
        result = minimize_scalar(huber_loss)
        return result.x if result.success else np.median(data)
    except:
        return np.median(data)

def extract_biweight_location(data, c=6.0, **kwargs):
    """Extract biweight location estimator (Tukey's biweight)"""
    try:
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return median
        
        u = (data - median) / (c * mad)
        weights = np.where(np.abs(u) < 1, (1 - u**2)**2, 0)
        
        if np.sum(weights) == 0:
            return median
        
        return np.sum(weights * data) / np.sum(weights)
    except:
        return np.median(data)

def extract_hodges_lehmann(data, **kwargs):
    """Extract Hodges-Lehmann estimator (median of pairwise averages)"""
    try:
        n = len(data)
        if n < 2:
            return np.mean(data)
        
        # For large datasets, use a sample to avoid memory issues
        if n > 1000:
            idx = np.random.choice(n, 1000, replace=False)
            data_sample = data[idx]
        else:
            data_sample = data
        
        pairwise_averages = []
        for i in range(len(data_sample)):
            for j in range(i, len(data_sample)):
                pairwise_averages.append((data_sample[i] + data_sample[j]) / 2)
        
        return np.median(pairwise_averages)
    except:
        return np.median(data)

def extract_qn_estimator(data, **kwargs):
    """Extract Qn robust scale estimator"""
    try:
        n = len(data)
        if n < 2:
            return np.std(data)
        
        # For large datasets, use a sample
        if n > 1000:
            idx = np.random.choice(n, 1000, replace=False)
            data_sample = data[idx]
        else:
            data_sample = data
        
        n = len(data_sample)
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                distances.append(abs(data_sample[i] - data_sample[j]))
        
        # Qn is the kth order statistic where k = choose(h,2) and h = floor(n/2) + 1
        h = n // 2 + 1
        k = int(h * (h - 1) / 2)
        
        if k <= len(distances):
            return np.sort(distances)[k-1] * 2.2219  # Consistency factor for normal distribution
        else:
            return np.std(data_sample)
    except:
        return np.std(data)

def extract_sn_estimator(data, **kwargs):
    """Extract Sn robust scale estimator"""
    try:
        n = len(data)
        if n < 2:
            return np.std(data)
        
        # For computational efficiency, sample for large datasets
        if n > 500:
            idx = np.random.choice(n, 500, replace=False)
            data_sample = data[idx]
        else:
            data_sample = data
        
        n = len(data_sample)
        medians = []
        
        for i in range(n):
            distances = np.abs(data_sample - data_sample[i])
            medians.append(np.median(distances))
        
        return np.median(medians) * 1.1926  # Consistency factor for normal distribution
    except:
        return np.std(data)

# ============================================================================
# ADVANCED DISTRIBUTION METRICS
# ============================================================================

def extract_entropy(data, bins=50, **kwargs):
    """Extract Shannon entropy"""
    try:
        hist, _ = np.histogram(data, bins=bins, density=True)
        # Avoid log(0) by adding small constant
        hist = hist + 1e-10
        bin_width = (np.max(data) - np.min(data)) / bins
        prob = hist * bin_width
        prob = prob / np.sum(prob)  # Ensure normalization
        
        return -np.sum(prob * np.log2(prob + 1e-10))
    except:
        return np.nan

def extract_differential_entropy(data, **kwargs):
    """Extract differential entropy using kernel density estimation"""
    try:
        from scipy.stats import gaussian_kde
        
        # Use KDE to estimate probability density
        kde = gaussian_kde(data)
        # Evaluate at data points
        densities = kde(data)
        
        # Remove zeros and very small values
        densities = densities[densities > 1e-10]
        
        if len(densities) == 0:
            return np.nan
        
        # Differential entropy
        return -np.mean(np.log(densities))
    except:
        return np.nan

def extract_gini_coefficient(data, **kwargs):
    """Extract Gini coefficient (inequality measure)"""
    try:
        # Ensure non-negative values
        data_pos = np.abs(data)
        sorted_data = np.sort(data_pos)
        n = len(sorted_data)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_data)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    except:
        return np.nan

def extract_moment_3(data, standardized=True, **kwargs):
    """Extract third moment (skewness)"""
    try:
        if standardized:
            return stats.skew(data)
        else:
            mean_val = np.mean(data)
            return np.mean((data - mean_val)**3)
    except:
        return np.nan

def extract_moment_4(data, standardized=True, **kwargs):
    """Extract fourth moment (kurtosis)"""
    try:
        if standardized:
            return stats.kurtosis(data, fisher=False)  # Pearson's definition (normal = 3)
        else:
            mean_val = np.mean(data)
            return np.mean((data - mean_val)**4)
    except:
        return np.nan

def extract_l_moment_3(data, **kwargs):
    """Extract L-moment 3"""
    try:
        sorted_data = np.sort(data)
        n = len(data)
        if n < 3:
            return np.nan
        
        l3 = 0
        for i in range(n):
            l3 += ((i-1)*i - (n-i-2)*(n-i-1)) * sorted_data[i]
        return l3 / (n * (n-1) * (n-2))
    except:
        return np.nan

def extract_l_moment_4(data, **kwargs):
    """Extract L-moment 4"""
    try:
        sorted_data = np.sort(data)
        n = len(data)
        if n < 4:
            return np.nan
        
        l4 = 0
        for i in range(n):
            term = (i-3)*(i-2)*(i-1) - 3*(i-2)*(i-1)*(n-i-1) + \
                   3*(i-1)*(n-i-2)*(n-i-1) - (n-i-3)*(n-i-2)*(n-i-1)
            l4 += term * sorted_data[i]
        
        return l4 / (n * (n-1) * (n-2) * (n-3))
    except:
        return np.nan

def extract_l_kurtosis(data, **kwargs):
    """Extract L-kurtosis (L4/L2)"""
    try:
        l2 = extract_l_moment_2(data)
        l4 = extract_l_moment_4(data)
        
        if l2 == 0 or l4 is None:
            return np.nan
        
        return l4 / l2
    except:
        return np.nan

# ============================================================================
# FRACTAL AND COMPLEXITY MEASURES
# ============================================================================

def extract_hurst_exponent(data, **kwargs):
    """Extract Hurst exponent using R/S analysis"""
    try:
        n = len(data)
        if n < 10:
            return np.nan
        
        # Calculate mean
        mean_val = np.mean(data)
        
        # Calculate cumulative deviations
        cumdev = np.cumsum(data - mean_val)
        
        # Calculate ranges for different window sizes
        lags = np.unique(np.logspace(1, np.log10(n/4), 10).astype(int))
        rs_values = []
        
        for lag in lags:
            if lag >= n:
                continue
                
            # Divide series into non-overlapping windows
            n_windows = n // lag
            rs_window = []
            
            for i in range(n_windows):
                start = i * lag
                end = start + lag
                
                window_cumdev = cumdev[start:end] - cumdev[start]
                window_data = data[start:end]
                
                if len(window_data) < 2:
                    continue
                
                # Range
                R = np.max(window_cumdev) - np.min(window_cumdev)
                
                # Standard deviation
                S = np.std(window_data)
                
                if S > 0:
                    rs_window.append(R / S)
            
            if rs_window:
                rs_values.append(np.mean(rs_window))
        
        if len(rs_values) < 3 or len(lags[:len(rs_values)]) < 3:
            return np.nan
        
        # Linear regression on log-log plot
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        # Remove infinite values
        valid_idx = np.isfinite(log_lags) & np.isfinite(log_rs)
        if np.sum(valid_idx) < 3:
            return np.nan
        
        hurst = np.polyfit(log_lags[valid_idx], log_rs[valid_idx], 1)[0]
        return hurst
    except:
        return np.nan

def extract_approximate_entropy(data, m=2, r=None, **kwargs):
    """Extract Approximate Entropy (ApEn)"""
    try:
        N = len(data)
        if N < m + 1:
            return np.nan
        
        if r is None:
            r = 0.2 * np.std(data)
        
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
            phi = 0.0
            
            for i in range(N - m + 1):
                template_i = patterns[i]
                matches = sum([1 for j in range(N - m + 1) 
                             if _maxdist(template_i, patterns[j]) <= r])
                if matches > 0:
                    phi += np.log(matches / (N - m + 1))
            
            return phi / (N - m + 1)
        
        return _phi(m) - _phi(m + 1)
    except:
        return np.nan

def extract_sample_entropy(data, m=2, r=None, **kwargs):
    """Extract Sample Entropy (SampEn)"""
    try:
        N = len(data)
        if N < m + 1:
            return np.nan
        
        if r is None:
            r = 0.2 * np.std(data)
        
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
            matches = 0
            
            for i in range(N - m):
                template_i = patterns[i]
                for j in range(i + 1, N - m + 1):
                    if _maxdist(template_i, patterns[j]) <= r:
                        matches += 1
            
            return matches
        
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        if phi_m == 0:
            return np.nan
        
        return -np.log(phi_m1 / phi_m)
    except:
        return np.nan

# ============================================================================
# SPECTRAL AND FREQUENCY DOMAIN FEATURES
# ============================================================================

def extract_spectral_centroid(data, **kwargs):
    """Extract spectral centroid (center of mass of spectrum)"""
    try:
        # Compute FFT
        fft_data = np.fft.fft(data)
        magnitude = np.abs(fft_data)
        
        # Frequency bins
        freqs = np.fft.fftfreq(len(data))
        
        # Only use positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        if np.sum(positive_magnitude) == 0:
            return np.nan
        
        # Spectral centroid
        centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
        return centroid
    except:
        return np.nan

def extract_spectral_spread(data, **kwargs):
    """Extract spectral spread (spread around spectral centroid)"""
    try:
        # Compute FFT
        fft_data = np.fft.fft(data)
        magnitude = np.abs(fft_data)
        
        # Frequency bins
        freqs = np.fft.fftfreq(len(data))
        
        # Only use positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        if np.sum(positive_magnitude) == 0:
            return np.nan
        
        # Spectral centroid
        centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
        
        # Spectral spread
        spread = np.sqrt(np.sum(((positive_freqs - centroid) ** 2) * positive_magnitude) / 
                        np.sum(positive_magnitude))
        return spread
    except:
        return np.nan

def extract_spectral_rolloff(data, roll_percent=85, **kwargs):
    """Extract spectral rolloff (frequency below which specified percentage of energy lies)"""
    try:
        # Compute FFT
        fft_data = np.fft.fft(data)
        magnitude = np.abs(fft_data)
        
        # Only use positive frequencies
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        # Cumulative energy
        cumsum_energy = np.cumsum(positive_magnitude)
        total_energy = cumsum_energy[-1]
        
        if total_energy == 0:
            return np.nan
        
        # Find rolloff point
        rolloff_point = roll_percent / 100 * total_energy
        rolloff_idx = np.where(cumsum_energy >= rolloff_point)[0]
        
        if len(rolloff_idx) == 0:
            return np.nan
        
        return rolloff_idx[0] / len(positive_magnitude)
    except:
        return np.nan

# ============================================================================
# EXTREME VALUE STATISTICS
# ============================================================================

def extract_gev_shape(data, **kwargs):
    """Extract shape parameter from Generalized Extreme Value distribution"""
    try:
        # Fit GEV distribution
        c, loc, scale = stats.genextreme.fit(data)
        return -c  # Convert to standard parameterization
    except:
        return np.nan

def extract_gev_scale(data, **kwargs):
    """Extract scale parameter from Generalized Extreme Value distribution"""
    try:
        c, loc, scale = stats.genextreme.fit(data)
        return scale
    except:
        return np.nan

def extract_gev_location(data, **kwargs):
    """Extract location parameter from Generalized Extreme Value distribution"""
    try:
        c, loc, scale = stats.genextreme.fit(data)
        return loc
    except:
        return np.nan

def extract_peaks_over_threshold(data, threshold=None, **kwargs):
    """Extract number of peaks over threshold (for extreme value analysis)"""
    try:
        if threshold is None:
            threshold = np.percentile(data, 95)
        
        return np.sum(data > threshold)
    except:
        return np.nan

# Example usage and function registry
FEATURE_FUNCTIONS = {
    # Basic robust statistics
    'median': extract_median,
    'iqr': extract_iqr,
    'mad': extract_mad,
    'p95': lambda data, **kwargs: extract_percentile(data, percentile=95, **kwargs),
    'p99': lambda data, **kwargs: extract_percentile(data, percentile=99, **kwargs),
    'p5': lambda data, **kwargs: extract_percentile(data, percentile=5, **kwargs),
    'p1': lambda data, **kwargs: extract_percentile(data, percentile=1, **kwargs),
    'cv': extract_cv,
    'trimmed_mean_10': lambda data, **kwargs: extract_trimmed_mean(data, trim_percent=10, **kwargs),
    'trimmed_mean_20': lambda data, **kwargs: extract_trimmed_mean(data, trim_percent=20, **kwargs),
    'range': extract_range,
    'quantile_skewness': extract_quantile_skewness,
    
    # L-moments
    'l_location': extract_l_moment_1,
    'l_scale': extract_l_moment_2,
    'l_skewness': extract_l_skewness,
    'l_moment_3': extract_l_moment_3,
    'l_moment_4': extract_l_moment_4,
    'l_kurtosis': extract_l_kurtosis,
    
    # Robust estimators
    'huber_location': extract_huber_location,
    'biweight_location': extract_biweight_location,
    'hodges_lehmann': extract_hodges_lehmann,
    'qn_estimator': extract_qn_estimator,
    'sn_estimator': extract_sn_estimator,
    
    # Advanced distribution metrics
    'entropy': extract_entropy,
    'differential_entropy': extract_differential_entropy,
    'gini_coefficient': extract_gini_coefficient,
    'moment_3': extract_moment_3,
    'moment_4': extract_moment_4,
    'standardized_moment_3': lambda data, **kwargs: extract_moment_3(data, standardized=True, **kwargs),
    'standardized_moment_4': lambda data, **kwargs: extract_moment_4(data, standardized=True, **kwargs),
    
    # Complexity and fractal measures
    'hurst_exponent': extract_hurst_exponent,
    'approximate_entropy': extract_approximate_entropy,
    'sample_entropy': extract_sample_entropy,
    
    # Spectral features
    'spectral_centroid': extract_spectral_centroid,
    'spectral_spread': extract_spectral_spread,
    'spectral_rolloff': extract_spectral_rolloff,
    'spectral_rolloff_95': lambda data, **kwargs: extract_spectral_rolloff(data, roll_percent=95, **kwargs),
    
    # Extreme value statistics
    'gev_shape': extract_gev_shape,
    'gev_scale': extract_gev_scale,
    'gev_location': extract_gev_location,
    'peaks_over_threshold': extract_peaks_over_threshold,
    
    # Threshold-based features
    'prop_above_median': extract_proportion_above_threshold,
    'prop_above_p75': lambda data, **kwargs: extract_proportion_above_threshold(data, threshold=np.percentile(data, 75), **kwargs),
    'prop_above_p90': lambda data, **kwargs: extract_proportion_above_threshold(data, threshold=np.percentile(data, 90), **kwargs),
    
    # Parametric distribution fits
    'gamma_shape': extract_gamma_shape,
    'gamma_scale': extract_gamma_scale,
    'lognormal_mu': extract_lognormal_mu,
    'lognormal_sigma': extract_lognormal_sigma,
    'normal_mean': extract_normal_mean,
    'normal_std': extract_normal_std,
    'exponential_rate': extract_exponential_rate,
}

def extract_all_features(data, feature_names=None, n_jobs=-1):
    """
    Extract all specified features from data using parallel processing
    
    Parameters:
    -----------
    data : array-like
        Peak-to-peak amplitude data
    feature_names : list, optional
        List of feature names to extract. If None, extracts all features.
    n_jobs : int, optional
        Number of parallel jobs (-1 uses all available cores)
    
    Returns:
    --------
    dict : Dictionary with feature names as keys and extracted values as values
    """
    
    if feature_names is None:
        feature_names = list(FEATURE_FUNCTIONS.keys())
    
    def extract_single_feature(name):
        if name in FEATURE_FUNCTIONS:
            try:
                return name, FEATURE_FUNCTIONS[name](data)
            except Exception as e:
                warnings.warn(f"Failed to extract feature '{name}': {e}")
                return name, np.nan
        else:
            warnings.warn(f"Unknown feature name: {name}")
            return name, np.nan
    
    # Parallel feature extraction
    results_list = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(extract_single_feature)(name) for name in feature_names
    )
    
    # Convert to dictionary
    results = dict(results_list)
    
    return results

