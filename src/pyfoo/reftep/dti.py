import numpy as np
import pandas as pd
import re

def read_trackvis_length_file(filename):
    """
    Read TrackVis length histogram file
    
    Parameters
    ----------
    filename : str
        Path to the TrackVis length file (e.g., *_Length.txt)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'metadata': dict with file info, track group, number of tracks, etc.
        - 'statistics': dict with mean vector, min/max/mean length
        - 'histogram': pandas DataFrame with length bins and counts
        - 'lengths': numpy array of length values
        - 'counts': numpy array of track counts
    """
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Initialize result dictionary
    result = {
        'metadata': {},
        'statistics': {},
        'histogram': None,
        'lengths': None,
        'counts': None
    }
    
    # Parse header information
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Track file path
        if line.startswith('Track file:'):
            result['metadata']['track_file'] = line.split(':', 1)[1].strip()
        
        # TrackVis version
        elif line.startswith('TrackVis'):
            version_match = re.search(r'Version\s+([\d.]+)', line)
            if version_match:
                result['metadata']['version'] = version_match.group(1)
        
        # Track group
        elif line.startswith('TrackGroup:'):
            i += 1
            result['metadata']['track_group'] = lines[i].strip()
        
        # Number of tracks
        elif line.startswith('Number of tracks:'):
            i += 1
            result['metadata']['num_tracks'] = int(lines[i].strip())
        
        # Mean Vector
        elif line.startswith('Mean Vector:'):
            i += 1
            vector_str = lines[i].strip().strip('()')
            result['statistics']['mean_vector'] = tuple(map(float, vector_str.split(',')))
        
        # Minimum Length
        elif line.startswith('Minimum Length:'):
            i += 1
            result['statistics']['min_length'] = float(lines[i].strip().split()[0])
        
        # Maximum Length
        elif line.startswith('Maximum Length:'):
            i += 1
            result['statistics']['max_length'] = float(lines[i].strip().split()[0])
        
        # Mean Length
        elif line.startswith('Mean Length:'):
            i += 1
            mean_str = lines[i].strip()
            parts = mean_str.split('+/-')
            result['statistics']['mean_length'] = float(parts[0].strip())
            result['statistics']['std_length'] = float(parts[1].strip().split()[0])
        
        # Number of points
        elif line.startswith('Number of points:'):
            result['statistics']['num_bins'] = int(line.split(':')[1].strip())
        
        # Length values
        elif line.startswith('Length (mm)'):
            i += 1
            length_values = []
            # Read length values until we hit "Track Count"
            while i < len(lines) and not lines[i].strip().startswith('Track Count'):
                values = lines[i].strip().split()
                length_values.extend([float(v) for v in values])
                i += 1
            result['lengths'] = np.array(length_values)
            continue  # Don't increment i again
        
        # Track Count
        elif line.startswith('Track Count'):
            i += 1
            count_values = []
            # Read all remaining count values
            while i < len(lines):
                if lines[i].strip():  # Skip empty lines
                    values = lines[i].strip().split()
                    count_values.extend([int(v) for v in values])
                i += 1
            result['counts'] = np.array(count_values)
            break  # We're done
        
        i += 1
    
    # Create histogram DataFrame
    if result['lengths'] is not None and result['counts'] is not None:
        result['histogram'] = pd.DataFrame({
            'length_mm': result['lengths'],
            'count': result['counts']
        })
    
    return result


def plot_track_length_histogram(data, title=None):
    """
    Plot the track length histogram
    
    Parameters
    ----------
    data : dict
        Output from read_trackvis_length_file()
    title : str, optional
        Custom title for the plot
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot histogram
    ax.bar(data['lengths'], data['counts'], 
           width=np.diff(data['lengths'])[0] * 0.9,
           color='steelblue', edgecolor='black', alpha=0.7)
    
    # Add statistics
    stats = data['statistics']
    stats_text = (
        f"Mean: {stats['mean_length']:.2f} ± {stats['std_length']:.2f} mm\n"
        f"Min: {stats['min_length']:.2f} mm\n"
        f"Max: {stats['max_length']:.2f} mm\n"
        f"Total tracks: {data['metadata']['num_tracks']}"
    )
    
    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    
    # Add vertical lines for mean
    ax.axvline(stats['mean_length'], color='red', linestyle='--', 
               linewidth=2, label='Mean', alpha=0.7)
    
    # Labels and formatting
    ax.set_xlabel('Track Length (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Track Count', fontsize=12, fontweight='bold')
    
    if title is None:
        title = f"Track Length Distribution - {data['metadata'].get('track_group', 'Unknown')}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig, ax


def summarize_track_data(data):
    """
    Print a summary of the track data
    
    Parameters
    ----------
    data : dict
        Output from read_trackvis_length_file()
    """
    print("=" * 60)
    print("TRACKVIS LENGTH DATA SUMMARY")
    print("=" * 60)
    
    print("\n--- Metadata ---")
    for key, value in data['metadata'].items():
        print(f"  {key}: {value}")
    
    print("\n--- Statistics ---")
    for key, value in data['statistics'].items():
        if isinstance(value, tuple):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")
    
    print("\n--- Histogram Summary ---")
    print(f"  Number of bins: {len(data['lengths'])}")
    print(f"  Total tracks: {np.sum(data['counts'])}")
    print(f"  Non-zero bins: {np.sum(data['counts'] > 0)}")
    print(f"  Mode (most frequent length): {data['lengths'][np.argmax(data['counts'])]:.2f} mm")
    print(f"  Tracks at mode: {np.max(data['counts'])}")
    
    # Calculate percentiles
    expanded_lengths = np.repeat(data['lengths'], data['counts'].astype(int))
    percentiles = [25, 50, 75]
    print("\n--- Percentiles ---")
    for p in percentiles:
        print(f"  {p}th percentile: {np.percentile(expanded_lengths, p):.2f} mm")
    
    print("=" * 60)


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Read the file
    filename = '/media/robbis/BLACK_16GB/dwi/sub-219_ses-mri1_tracks_10M-Track_1_Length.txt'
    data = read_trackvis_length_file(filename)
    
    # Print summary
    summarize_track_data(data)
    
    # Access specific data
    print("\n--- Quick Access Examples ---")
    print(f"Mean length: {data['statistics']['mean_length']:.2f} mm")
    print(f"Number of tracks: {data['metadata']['num_tracks']}")
    print(f"First 5 length bins: {data['lengths'][:5]}")
    print(f"First 5 counts: {data['counts'][:5]}")
    
    # Plot the histogram
    fig, ax = plot_track_length_histogram(data)
    plt.savefig('track_length_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Access as DataFrame
    print("\n--- Histogram DataFrame ---")
    print(data['histogram'].head(10))
    
    # Calculate cumulative distribution
    data['histogram']['cumulative_count'] = data['histogram']['count'].cumsum()
    data['histogram']['cumulative_percent'] = (
        data['histogram']['cumulative_count'] / data['histogram']['count'].sum() * 100
    )
    
    print("\n--- With Cumulative Distribution ---")
    print(data['histogram'].head(10))