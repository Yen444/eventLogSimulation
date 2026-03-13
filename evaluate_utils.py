"""
Utility functions for event log evaluation and analysis.

This module contains commonly used functions for:
- Data preprocessing and filtering
- Visualization and plotting  
- Statistical analysis
- General utility operations

Extracted from evaluate.py to improve code organization and reusability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import re
from datetime import timedelta
from typing import List, Dict, Any
import pm4py
from pm4py.statistics.variants.log import get as variants_get

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np
from itertools import product
import distance
import ot


def filter_complete_processes(tiny_db_json_path):
    """Return a list of process_id strings for processes marked as completed.

    This function is robust to multiple JSON shapes produced by the simulation
    store or TinyDB exports. Supported shapes:
    - {'processes': { '<doc_key>': { 'process_id': ..., 'completed': true }, ... }}
    - {'processes': [ { 'process_id': ..., 'completed': true }, ... ]}

    If the file can't be read or no completed processes are found an empty list
    is returned.
    """
    try:
        # First try UTF-8 encoding
        with open(tiny_db_json_path, 'r', encoding='utf-8') as fh:
            db = json.load(fh)
            print(f'successfully loaded simulation store from {tiny_db_json_path}')
    except UnicodeDecodeError:
        # If UTF-8 fails, try with latin-1 encoding which accepts all bytes
        try:
            with open(tiny_db_json_path, 'r', encoding='latin-1') as fh:
                db = json.load(fh)
                print(f'successfully loaded simulation store from {tiny_db_json_path} (using latin-1 encoding)')
        except Exception as e:
            print(f'could not load simulation store from {tiny_db_json_path}: {e}')
            return []
    except Exception as e:
        print(f'could not load simulation store from {tiny_db_json_path}: {e}')
        return []

    # Extract the processes table from a dict-shaped DB
    processes = None
    if isinstance(db, dict):
        # common key
        processes = db.get('processes')
        # Some TinyDB exports nest tables under a 'tables' key or similar; try a couple
        if processes is None:
            # look for keys that equal 'processes' ignoring case
            for k, v in db.items():
                if k.lower() == 'processes' and v:
                    processes = v
                    break

    # If processes is still None, nothing to do
    if processes is None:
        print('no processes found in the simulation store')
        return []

    completed_ids = []

    # processes can be either a dict mapping -> record, or a list of records
    if isinstance(processes, dict):
        iterable = processes.values()
    elif isinstance(processes, list):
        iterable = processes
    else:
        # unsupported shape
        print('unsupported processes structure in the simulation store')
        return []
    print(f'found {len(processes)} processes in the simulation store')
    for proc in iterable:
        if not isinstance(proc, dict):
            print('skipping invalid process record in the simulation store')
            continue
        # completed may be missing; treat truthy True only
        if proc.get('completed') is True:
            pid = proc.get('process_id') or proc.get('id')
            if pid is not None:
                completed_ids.append(pid)

    return completed_ids


def parse_dict_string(s: str) -> Dict[str, Any]:
    """Parse strings like 'amount: 20, time: 25' into dicts."""
    if pd.isna(s) or not isinstance(s, str) or s.strip() == "":
        return {}
    # Convert to dictionary format safely
    try:
        items = [x.strip() for x in s.split(",")]
        d = {}
        for item in items:
            if ":" in item:
                k, v = item.split(":", 1)
                v = v.strip()
                if v == "":
                    continue
                # Try numeric conversion
                try:
                    v = float(v) if "." in v else int(v)
                except ValueError:
                    pass
                d[k.strip()] = v
        return d
    except Exception:
        return {}


def merge_dicts(dicts):
    """Merge a list of dictionaries into one."""
    result = {}
    for d in dicts:
        result.update(d)  # or handle key conflicts differently if needed
    return result


def extract_gender(agent_data):
    """Extract gender from agent_data string."""
    if pd.isna(agent_data):
        return None
    match = re.search(r'gender:\s*(Male|Female)', agent_data)
    return match.group(1) if match else None

def plot_numeric_distributions(gen_df, gen_attr, real_df, real_attr, bins=20, title=None, outfile=None, figsize=(8, 5)):
    """
    Plots overlapping histograms and KDEs for two numeric columns using matplotlib.
    """
    # Convert to numeric
    data_gen = pd.to_numeric(gen_df[gen_attr], errors='coerce').dropna()
    data_real = pd.to_numeric(real_df[real_attr], errors='coerce').dropna()

    plt.figure(figsize=figsize)
    plt.hist(data_gen, bins=bins, alpha=0.5, color='blue', density=True, label='Generated Data')
    plt.hist(data_real, bins=bins, alpha=0.5, color='red', density=True, label='Real Data')
    plt.grid(True, alpha=0.3)
    
    # Optional: KDE approximation using smooth curve
    try:
        from scipy.stats import gaussian_kde
        kde_gen = gaussian_kde(data_gen)
        kde_real = gaussian_kde(data_real)
        x_grid = np.linspace(min(data_gen.min(), data_real.min()),
                             max(data_gen.max(), data_real.max()), 1000)
        plt.plot(x_grid, kde_gen(x_grid), color='blue')
        plt.plot(x_grid, kde_real(x_grid), color='red')
    except ImportError:
        pass  # skip KDE if scipy not available
    
    plt.xlabel(f'Net worth')
    plt.ylabel('Density')
    plt.title(title if title else f'Distribution of generated and real')
    plt.legend()
    
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_numeric_distribution(df, attr, bins=20, color='blue', label='Data', title=None, outfile=None, figsize=(5, 4)):
    """
    Plots histogram and KDE for a single numeric column using matplotlib.
    """
    # Convert to numeric
    data = pd.to_numeric(df[attr], errors='coerce').dropna()
    
    if data.empty:
        print(f"No valid numeric data found for attribute '{attr}'")
        return

    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, alpha=0.7, color=color, density=True, label=label)
    plt.grid(True, alpha=0.3)
    
    # Optional: KDE approximation using smooth curve
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_grid = np.linspace(data.min(), data.max(), 1000)
        plt.plot(x_grid, kde(x_grid), color=color, linewidth=2)
    except ImportError:
        pass  # skip KDE if scipy not available
    
    plt.xlabel(attr)
    plt.ylabel('Density')
    plt.title(title if title else f'Distribution of {attr}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_string_distributions(gen_df, gen_attr, real_df, real_attr, top_n=10, title=None, outfile=None, figsize=(10, 5)):
    """
    Plots overlapping bar charts for two string/categorical columns using matplotlib.
    """
    freq_gen = gen_df[gen_attr].value_counts(normalize=True).head(top_n)
    freq_real = real_df[real_attr].value_counts(normalize=True).head(top_n)
    
    all_categories = list(set(freq_gen.index) | set(freq_real.index))
    freq_gen_aligned = [freq_gen.get(cat, 0) for cat in all_categories]
    freq_real_aligned = [freq_real.get(cat, 0) for cat in all_categories]
    
    x = range(len(all_categories))
    plt.figure(figsize=figsize)
    plt.bar([i-0.2 for i in x], freq_gen_aligned, width=0.4, color='blue', alpha=0.7, label='Generated Data')
    plt.bar([i+0.2 for i in x], freq_real_aligned, width=0.4, color='red', alpha=0.7, label='Real Data')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(x, all_categories, rotation=45, ha='right')
    plt.ylabel('Proportion')
    plt.title(title if title else f'Distribution of {gen_attr} (Generated) and {real_attr} (Real)')
    plt.legend()
    plt.tight_layout()
    
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_string_distribution(df, attr, top_n=10, color='blue', label='Data', title=None, normalize=True, outfile=None, figsize=(5, 4)):
    """
    Plots bar chart for a single string/categorical column using matplotlib.
    
    Parameters:
    - df: DataFrame containing the data
    - attr: column name to plot
    - top_n: number of top categories to show
    - color: bar color
    - label: label for the data
    - normalize: if True, show proportions; if False, show counts
    - outfile: optional file path to save the plot
    - figsize: tuple specifying figure size (width, height)
    """
    if attr not in df.columns:
        print(f"Column '{attr}' not found in DataFrame")
        return
    
    # Get value counts
    freq = df[attr].value_counts(normalize=normalize).head(top_n)
    
    if freq.empty:
        print(f"No data found for attribute '{attr}'")
        return
    
    categories = freq.index.tolist()
    values = freq.values.tolist()
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(categories)), values, color=color, alpha=0.7, label=label)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if normalize:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
    
    plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
    plt.ylabel('Proportion' if normalize else 'Count')
    plt.title(title if title else f'Distribution of {attr}')
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3, axis='y')
    
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def generate_process_model(log_df, case_id_col, activity_col, timestamp_col, noise_threshold, out_file=None):
    """
    Generate a process model from the log DataFrame.

    Parameters:
        log_df: pandas DataFrame containing the event log.
        case_id_col: column name for case IDs.
        activity_col: column name for activity names.
        timestamp_col: column name for timestamps.
        noise_threshold: noise threshold for process discovery.
        out_file: optional file path to save the process model.

    Returns:
        None
    """
    process_model = pm4py.discover_bpmn_inductive(
        log_df,
        noise_threshold=noise_threshold,
        activity_key=activity_col,
        case_id_key=case_id_col,
        timestamp_key=timestamp_col)
    if out_file:
        pm4py.write_bpmn(process_model, out_file, auto_layout=True)

def trace_variants(log_df=None, case_id_col=None, activity_col=None, timestamp_col=None, 
                   xes_file=None, out_file=None):
    """
    Get the trace variants from either a log DataFrame or XES file.

    Parameters:
        log_df: pandas DataFrame containing the event log (optional if xes_file is provided).
        case_id_col: column name for case IDs (required if log_df is provided).
        activity_col: column name for activity names (required if log_df is provided).
        timestamp_col: column name for timestamps (required if log_df is provided).
        xes_file: path to the XES log file (optional if log_df is provided).
        out_file: optional file path to save the variants.

    Returns:
        dict: trace variants with frequencies
    """
    
    # Input validation
    if log_df is None and xes_file is None:
        raise ValueError("Either log_df or xes_file must be provided")
    
    if log_df is not None and xes_file is not None:
        raise ValueError("Provide either log_df or xes_file, not both")
    
    if log_df is not None and (case_id_col is None or activity_col is None or timestamp_col is None):
        raise ValueError("case_id_col, activity_col, and timestamp_col must be provided when using log_df")
    
    # Process XES file
    if xes_file is not None:
        log = pm4py.read_xes(xes_file)
        
        # Lower activity names
        for trace in log:
            for event in trace:
                if 'concept:name' in event:
                    event['concept:name'] = event['concept:name'].lower()
        
        variants = variants_get.get_variants(log)
    
    # Process DataFrame
    else:
        # Create a copy to avoid modifying the original
        log_df_copy = log_df.copy()
        
        log_df_copy[timestamp_col] = pd.to_datetime(
            log_df_copy[timestamp_col],
            utc=True,
            format='mixed'  # to solve the mixed format issue
        )
        log_df_copy = log_df_copy.dropna(subset=[timestamp_col, case_id_col])
        # lower activity names
        log_df_copy[activity_col] = log_df_copy[activity_col].str.lower()

        variants = pm4py.get_variants(
            log_df_copy,
            activity_key=activity_col,
            case_id_key=case_id_col,
            timestamp_key=timestamp_col
        )

        log_df_copy = log_df_copy.rename(columns={
            case_id_col: "case:concept:name",
            activity_col: "concept:name",
            timestamp_col: "time:timestamp"
        })   
        variants = variants_get.get_variants(log=log_df_copy)
    
    # Common processing for both input types
    # print number of variants
    print(len(variants), 'variants found')
    # print average len of variants
    avg_len = sum(len(v) for v in variants.keys()) / len(variants.keys()) if variants else 0
    print(f"Average length of variants: {avg_len:.2f}")
    
    # write all variants ordered by frequency (descending)
    variant_dict = {}
    # sort variants by their frequency (number of traces) descending
    sorted_variants = sorted(variants.items(), key=lambda item: len(item[1]), reverse=True)
    
    if out_file:
        with open(out_file, 'w', encoding='utf-8') as f:
            for idx, (variant, traces) in enumerate(sorted_variants, start=1):
                freq = len(traces)
                f.write(f"Variant #{idx}: {variant}\n")
                for trace in traces:
                    f.write(f"  Trace: {trace}\n")
                f.write(f"  Frequency: {freq}\n\n")
                variant_dict[variant] = freq
    else:
        for idx, (variant, traces) in enumerate(sorted_variants, start=1):
            freq = len(traces)
            variant_dict[variant] = freq
    
    return variant_dict



def visualize_logs_2d(logs: dict, method: str = "tsne", perplexity: int = 30, random_state: int = 42, title: str = None, outfile: str = None) -> dict:
    """
    Visualize multiple process logs (tuple-variant dicts) in 2D using fixed-length integer encoding.
    Shorter traces are padded with 0s to the length of the longest trace.
    
    Args:
        logs (dict): {log_name: {variant_tuple: frequency}}
        method (str): "tsne" or "pca"
        perplexity (int): t-SNE parameter
        random_state (int): random seed
    
    Returns:
        dict: activity mapping, reduced vectors, labels
    """

    # Build set of unique activities
    all_variants = []
    labels = []
    for log_name, log_data in logs.items():
        for variant, freq in log_data.items():
            # Lower all activities in the variant tuple
            lowered_variant = tuple(activity.lower() for activity in variant)
            all_variants.extend([lowered_variant] * freq)
            labels.extend([log_name] * freq)
    
    activities = sorted({a for trace in all_variants for a in trace})
    activity_to_id = {a: i+1 for i, a in enumerate(activities)}  # map activities 1..N

    print('All activities and their mappings:')
    for activity, idx in activity_to_id.items():
        print(f"  {activity}: {idx}")
    
    # Encode traces to fixed-length vectors
    max_len = max(len(trace) for trace in all_variants)

    def encode_trace(trace, mapping, max_len):
        vec = [mapping[a] for a in trace]          # map each activity to integer
        vec += [0] * (max_len - len(trace))       # pad with 0s
        return vec

    X = np.array([encode_trace(t, activity_to_id, max_len) for t in all_variants])
    labels = np.array(labels)

    # Scale and reduce dimensions
    X_scaled = StandardScaler().fit_transform(X)
    if method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    else:
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    
    X_reduced = reducer.fit_transform(X_scaled)

    # Filter out points with y values > 10 and x values < 2 (only for PCA)
    if method.lower() == "pca":
        #filter_mask = (X_reduced[:, 1] <= 5) & (X_reduced[:, 0] <= 0.5)
        filter_mask = np.ones(len(X_reduced), dtype=bool)  # No filtering
        X_reduced_filtered = X_reduced[filter_mask]
        labels_filtered = labels[filter_mask]
    else:
        X_reduced_filtered = X_reduced
        labels_filtered = labels
    
    # Plot
    plt.figure(figsize=(8,6))

    contrast_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'] 
    unique_logs = np.unique(labels_filtered)

    for i, log_name in enumerate(unique_logs):
        mask = labels_filtered == log_name
        plt.scatter(
            X_reduced_filtered[mask, 0],
            X_reduced_filtered[mask, 1],
            color=contrast_colors[i % len(contrast_colors)],
            alpha=0.4,
            s=50,
            edgecolor='k',   # black edge to make dots distinct
            linewidth=0.5,
            label=log_name
        )
    
    plt.title(title if title else "2D Visualization of Process Logs")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()
    
    plt.show()

        
    return {
        "activity_mapping": activity_to_id,
        "reduced_vectors": X_reduced_filtered,
        "labels": labels_filtered
    }
def normalized_damerau_levenshtein(trace1, trace2):
    """Compute normalized Damerau-Levenshtein distance between two traces."""
    if not trace1 and not trace2:
        return 0.0
    dist = distance.levenshtein(trace1, trace2)
    max_len = max(len(trace1), len(trace2))
    return dist / max_len if max_len > 0 else 0.0

def calculate_trace_variant_distribution_distance(log1_variants_dict: dict, log2_variants_dict: dict) -> float:
    """
    Calculate the distance between two trace variant distributions using Jensen-Shannon divergence.

    Args:
        log1_variants (dict): {variant_tuple: frequency} for log 1
        log2_variants (dict): {variant_tuple: frequency} for log 2

    Returns:
        float: Earth Mover's Distance between the two distributions
    """
    # create distance matrix using Levenshtein distance
    log1_variants = list(log1_variants_dict.keys())
    log2_variants = list(log2_variants_dict.keys())
    distance_matrix = np.zeros((len(log1_variants), len(log2_variants)))

    for i, v1 in enumerate(log1_variants):
        for j, v2 in enumerate(log2_variants):
            distance_matrix[i, j] = normalized_damerau_levenshtein(v1, v2)
    # create probability distributions
    total1 = sum(log1_variants_dict.values())
    total2 = sum(log2_variants_dict.values())
    prob_dist1 = np.array([log1_variants_dict[v] / total1 for v in log1_variants])
    prob_dist2 = np.array([log2_variants_dict[v] / total2 for v in log2_variants])
    # compute Earth Mover's Distance
    T = ot.emd(prob_dist1, prob_dist2, distance_matrix)
    emd = np.sum(T * distance_matrix)
    return emd
    
def main():
    #log1_variants = {('a', 'b', 'c'): 1, ('a', 'c', 'd'): 10, ('b', 'c'): 1}
    #log2_variants = {('a', 'b', 'c'): 1}
    #distance = calculate_trace_variant_distribution_distance(log1_variants, log2_variants)
    #print(distance)
    pass

if __name__ == "__main__":
    main()

