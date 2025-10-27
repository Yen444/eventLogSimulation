import pandas as pd
import math
from log_distance_measures.config import EventLogIDs
import numpy as np
from datetime import timedelta
import pm4py
from pm4py.statistics.variants.log import get as variants_get

import pandas as pd
import matplotlib.pyplot as plt
import re

import ast
from typing import List, Dict, Any

def filter_complete_processes(tiny_db_json_path):
    """Return a list of process_id strings for processes marked as completed.

    This function is robust to multiple JSON shapes produced by the simulation
    store or TinyDB exports. Supported shapes:
    - {'processes': { '<doc_key>': { 'process_id': ..., 'completed': true }, ... }}
    - {'processes': [ { 'process_id': ..., 'completed': true }, ... ]}

    If the file can't be read or no completed processes are found an empty list
    is returned.
    """
    import json
    try:
        with open(tiny_db_json_path, 'r', encoding='utf-8') as fh:
            db = json.load(fh)
    except Exception:
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
        return []

    completed_ids = []

    # processes can be either a dict mapping -> record, or a list of records
    if isinstance(processes, dict):
        iterable = processes.values()
    elif isinstance(processes, list):
        iterable = processes
    else:
        # unsupported shape
        return []

    for proc in iterable:
        if not isinstance(proc, dict):
            continue
        # completed may be missing; treat truthy True only
        if proc.get('completed') is True:
            pid = proc.get('process_id') or proc.get('id')
            if pid is not None:
                completed_ids.append(pid)

    return completed_ids

def check_characteristics(event_log: pd.DataFrame, outfile=None):
    """
    Characteristic 1: "Female employees take more time to complete tasks than male employees."
    Visualize with a boxplot using matplotlib only.
    """

    # Convert timestamps
    event_log['start_timestamp'] = pd.to_datetime(event_log['start_time'])
    event_log['end_timestamp'] = pd.to_datetime(event_log['end_time'])

    # Compute duration in seconds
    #event_log['duration'] = (event_log['end_timestamp'] - event_log['start_timestamp']).dt.total_seconds()
    # duration is in task_duration inside event_data column, in hours (string)
    event_log['duration'] = event_log['event_data'].str.extract(r'task_duration:\s*(\d+\.?\d*)').astype(float) * 3600  # convert hours to seconds
    # Extract gender from agent_data (example: "name: Klaas Jansen, age: 42, gender: Male")
    def extract_gender(agent_data):
        if pd.isna(agent_data):
            return None
        match = re.search(r'gender:\s*(Male|Female)', agent_data)
        return match.group(1) if match else None
    
    event_log['gender'] = event_log['Resource'].apply(extract_gender)

    # Drop rows without duration or gender
    df = event_log.dropna(subset=['duration', 'gender'])

    # Prepare data for boxplot
    male_durations = df[df['gender'] == 'Male']['duration']
    female_durations = df[df['gender'] == 'Female']['duration']

    # Boxplot
    plt.figure(figsize=(8,6))
    plt.boxplot([male_durations, female_durations], tick_labels=['Male', 'Female'])
    plt.title('Task Duration by Gender')
    plt.ylabel('Duration (seconds)')
    plt.xlabel('Gender')
    
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

    # Print mean durations for reference
    mean_duration = df.groupby('gender')['duration'].mean()
    print("Mean duration by gender:\n", mean_duration)


    # --- Characteristic 2: Senior employees for high networth SRM ---

def check_attribute_variability(log_df: pd.DataFrame, case_id_col: str, case_attrs: List[str], event_attrs: List[str]) -> Dict[str, Any]:
    """
    Check variability of attributes across cases and events for real logs.
    
    Assumes:
      - log_df is an event-level DataFrame
      - case and event attributes are directly available as columns
    
    Parameters:
        log_df: pandas DataFrame with event log.
        case_id_col: column identifying each case.
        case_attrs: list of case-level attribute names.
        event_attrs: list of event-level attribute names.

    Returns:
        A dictionary with variability statistics for each attribute.
    """
    variability_stats = {}

    # --- Case-level attributes (one value per case) ---
    # Get unique case-level data by grouping by case_id and taking first occurrence
    case_level_df = log_df.groupby(case_id_col).first().reset_index()

    for attr in case_attrs:
        if attr not in log_df.columns:
            continue
        
        series = case_level_df[attr].dropna()
        if series.empty:
            continue
        
        # Compute top frequencies
        value_counts = series.value_counts()
        top_n = value_counts.head(10)
        top_values = list(zip(top_n.index.tolist(), top_n.values.tolist()))

        stats_entry = {
            "level": "case",
            "unique_values": int(value_counts.shape[0]) if hasattr(value_counts, 'shape') else int(value_counts.size),
            "total_values": len(series),  # Total count of all non-NaN values
            "top_values": top_values,
            "freq_dict": value_counts.to_dict(),  # Full frequency dictionary for histogram plotting
        }

        # If numeric, also include basic numeric stats
        if pd.api.types.is_numeric_dtype(series):
            stats_entry.update({
                "mean": float(series.mean()),
                "max": float(series.max()),
                "min": float(series.min()),
                "std": float(series.std()),
            })

        variability_stats[attr] = stats_entry

    # --- Event-level attributes (evaluate per event) ---
    for attr in event_attrs:
        if attr not in log_df.columns:
            continue

        series = log_df[attr].dropna()
        if series.empty:
            continue
        
        # Compute top frequencies for event-level attributes
        value_counts = series.value_counts()
        top_n = value_counts.head(10)
        top_values = list(zip(top_n.index.tolist(), top_n.values.tolist()))

        stats_entry = {
            "level": "event",
            "unique_values": int(value_counts.shape[0]) if hasattr(value_counts, 'shape') else int(value_counts.size),
            "total_values": len(series),  # Total count of all non-NaN values
            "top_values": top_values,
            "freq_dict": value_counts.to_dict(),  # Full frequency dictionary for histogram plotting
        }

        if pd.api.types.is_numeric_dtype(series):
            stats_entry.update({
                "mean": float(series.mean()),
                "max": float(series.max()),
                "min": float(series.min()),
                "std": float(series.std()),
            })

        variability_stats[attr] = stats_entry
    
    return variability_stats

def pretty_print_attribute_variability(variability: Dict[str, Any], max_values: int = 10):
    """Nicely print attribute variability produced by check_attribute_variability.

    - Numeric attributes: prints mean, min, max, std, unique count.
    - Categorical attributes: prints top values with counts and percentages.
    """
    if not variability:
        print("No attribute variability information to display.")
        return

    # Determine column width for attribute names
    max_attr_len = max((len(str(k)) for k in variability.keys()), default=10)

    # Group attributes by level so we print all case-level first, then event-level
    case_attrs = [k for k, v in variability.items() if v.get('level') == 'case']
    event_attrs = [k for k, v in variability.items() if v.get('level') == 'event']
    other_attrs = [k for k in variability.keys() if k not in case_attrs and k not in event_attrs]

    ordered = sorted(case_attrs) + sorted(event_attrs) + sorted(other_attrs)

    for attr in ordered:
        stats = variability[attr]
        level = stats.get('level', 'unknown')
        header = f"Attribute: {attr} (level: {level})"
        print("\n" + header)
        print('-' * len(header))

        # Numeric-style stats
        if any(k in stats for k in ('mean', 'min', 'max', 'std')):
            mean = stats.get('mean')
            mn = stats.get('min')
            mx = stats.get('max')
            std = stats.get('std')
            
            if mean is not None:
                print(f"  mean: {mean:.4f}")
            if mn is not None:
                print(f"  min:  {mn}")
            if mx is not None:
                print(f"  max:  {mx}")
            if std is not None and (isinstance(std, (int, float)) and not math.isnan(std)):
                print(f"  std:  {std:.4f}")
        unique = stats.get('unique_values')
        if unique is not None:
            print(f"  unique values: {unique}")
        total_vals = stats.get('total_values')
        if total_vals is not None:
            print(f"  total values: {total_vals}")
        # Show top values when available (works for numeric and categorical)
        if 'top_values' in stats and stats.get('top_values'):
            pairs = stats.get('top_values')
            total = sum(cnt for _, cnt in pairs) if pairs else 1
            # If we computed top_values from value_counts.head(10), total should be <= total events
            # But for percentages we prefer the sum of all counts if available via 'unique_values' and 'top_values' lengths don't provide total:
            # Try to extract a better total if possible: if 'frequencies_total' provided, use it; otherwise use sum of all known counts in cases where stats include counts for all unique values.
            # (Simpler approach: compute percentages relative to the sum of all top_values counts — still informative.)
            print(f"  top {len(pairs)} values:")
            for val, cnt in pairs:
                pct = 100.0 * cnt / total_vals if total_vals else 0.0
                # format value as string, limit length
                val_str = (str(val))
                print(f"    {val_str:>{max_attr_len}} : {cnt:>6d} ({pct:5.2f}%)")
            if stats.get('unique_values', 0) and stats.get('unique_values') > len(pairs):
                remaining = stats.get('unique_values') - len(pairs)
                print(f"    ... and {remaining} more distinct values")
        elif 'unique_values' in stats:
            # Fallback: list (trimmed)
            uniques = stats.get('unique_values') or []
            display = uniques[:max_values]
            print(f"  distinct values: {len(uniques)} (showing up to {max_values})")
            for v in display:
                print(f"    - {v}")
            if len(uniques) > max_values:
                print(f"    ... and {len(uniques) - max_values} more values")
        else:
            # Fallback: print the dict
            print(f"  Stats: {stats}")



# Numeric columns (handles floats as strings)
def plot_numeric_distributions(gen_df, gen_attr, real_df, real_attr, bins=20, outfile=None):
    """
    Plots overlapping histograms and KDEs for two numeric columns using matplotlib.
    """
    # Convert to numeric
    data_gen = pd.to_numeric(gen_df[gen_attr], errors='coerce').dropna()
    data_real = pd.to_numeric(real_df[real_attr], errors='coerce').dropna()
    
    plt.figure(figsize=(8,5))
    plt.hist(data_gen, bins=bins, alpha=0.5, color='blue', density=True, label='Generated Data')
    plt.hist(data_real, bins=bins, alpha=0.5, color='red', density=True, label='Real Data')
    
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
    
    plt.xlabel(f'{gen_attr} / {real_attr}')
    plt.ylabel('Density')
    plt.title(f'Distribution of {gen_attr} (Generated) and {real_attr} (Real)')
    plt.legend()
    
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_numeric_distribution(df, attr, bins=20, color='blue', label='Data', outfile=None):
    """
    Plots histogram and KDE for a single numeric column using matplotlib.
    """
    # Convert to numeric
    data = pd.to_numeric(df[attr], errors='coerce').dropna()
    
    if data.empty:
        print(f"No valid numeric data found for attribute '{attr}'")
        return
    
    plt.figure(figsize=(8,5))
    plt.hist(data, bins=bins, alpha=0.7, color=color, density=True, label=label)
    
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
    plt.title(f'Distribution of {attr}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

# String/categorical columns
def plot_string_distributions(gen_df, gen_attr, real_df, real_attr, top_n=10, outfile=None):
    """
    Plots overlapping bar charts for two string/categorical columns using matplotlib.
    """
    freq_gen = gen_df[gen_attr].value_counts(normalize=True).head(top_n)
    freq_real = real_df[real_attr].value_counts(normalize=True).head(top_n)
    
    all_categories = list(set(freq_gen.index) | set(freq_real.index))
    freq_gen_aligned = [freq_gen.get(cat, 0) for cat in all_categories]
    freq_real_aligned = [freq_real.get(cat, 0) for cat in all_categories]
    
    x = range(len(all_categories))
    plt.figure(figsize=(10,5))
    plt.bar([i-0.2 for i in x], freq_gen_aligned, width=0.4, color='blue', alpha=0.7, label='Generated Data')
    plt.bar([i+0.2 for i in x], freq_real_aligned, width=0.4, color='red', alpha=0.7, label='Real Data')
    
    plt.xticks(x, all_categories, rotation=45, ha='right')
    plt.ylabel('Proportion')
    plt.title(f'Distribution of {gen_attr} (Generated) and {real_attr} (Real)')
    plt.legend()
    plt.tight_layout()
    
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_string_distribution(df, attr, top_n=10, color='blue', label='Data', normalize=True, outfile=None):
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
    
    plt.figure(figsize=(10, 6))
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
    plt.title(f'Distribution of {attr}')
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


def trace_variants(log_df, case_id_col, activity_col, timestamp_col, out_file=None):
    """
    Get the trace variants from the GLog CSV file.

    Parameters:
        in_file: path the input CSV file.

    Returns:
        trace variants
    """
    log_df[timestamp_col] = pd.to_datetime(
        log_df[timestamp_col],
        utc=True,
        format='mixed'  # to sovlve the mixed format issue
    )
    log_df = log_df.dropna(subset=[timestamp_col, case_id_col])
    # lower activity names
    log_df[activity_col] = log_df[activity_col].str.lower()

    variants = pm4py.get_variants(
        log_df,
        activity_key=activity_col,
        case_id_key=case_id_col,
        timestamp_key=timestamp_col
    )

    log_df = log_df.rename(columns={
        case_id_col: "case:concept:name",
        activity_col: "concept:name",
        timestamp_col: "time:timestamp"
    })   
    variants = variants_get.get_variants(log=log_df)
    # print number of variants
    print(len(variants), 'variants found')
    # print average len of variants
    avg_len = sum(len(v) for v in variants.keys()) / len(variants.keys()) if variants else 0
    print(f"Average length of variants: {avg_len:.2f}")
    # write all variants ordered by frequency (descending)
    if out_file:
        # sort variants by their frequency (number of traces) descending
        sorted_variants = sorted(variants.items(), key=lambda item: len(item[1]), reverse=True)
        with open(out_file, 'w', encoding='utf-8') as f:
            for idx, (variant, traces) in enumerate(sorted_variants, start=1):
                freq = len(traces)
                f.write(f"Variant #{idx}: {variant}\n")
                for trace in traces:
                    f.write(f"  Trace: {trace}\n")
                f.write(f"  Frequency: {freq}\n\n")

def preprocess_glog(in_file, tiny_db_json_path, num_traces=692):
    """Preprocess a generated log CSV into a normalized DataFrame.

    Steps:
    - Try reading with header; if header missing expected columns, fall back to headerless read with defaults
    - Normalize column names to 'case_id', 'Activity', 'end_time', 'Resource'
    - Parse timestamps to UTC and drop invalid rows
    - Select the first `num_traces` cases ordered by case start time
    - Lowercase activity names and sort events by case and timestamp

    Returns a pandas.DataFrame with columns at least: 'case_id', 'Activity', 'end_time', 'Resource'.
    """
    # Get completed processes id only
    completed_process_ids = set(filter_complete_processes(tiny_db_json_path))
    if len(completed_process_ids) < num_traces:
        num_traces = len(completed_process_ids)
    
    default_col_names = ['iteration_id', 'case_id', 'Activity', 'agent_id', 'start_time', 'end_time', 'role', 'resource', 'case_data', 'event_data', 'message', 'end', 'process_state']
    # try reading with header, fall back to headerless
    try:
        glog = pd.read_csv(in_file, header=0)
        if not any(c in glog.columns for c in ['case_id', 'case:concept:name']):
            raise ValueError('header missing expected case id')
    except Exception:
        glog = pd.read_csv(in_file, header=None, names=default_col_names)

    # PARSE AND MERGE EVENT/CASE DICTIONARIES BY CASE, CREATING UNIFIED CASE-LEVEL ATTRIBUTE COLUMNS
    # column case_data and event_data contain string values of dict {atr:val}
    # parse those dict strings
    # merge the dicts of different rows of the same case_id into one
    # all rows of same case_id will have the same values for those attributes

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
        result = {}
        for d in dicts:
            result.update(d)  # or handle key conflicts differently if needed
        return result
    
    # Parse case_data and event_data columns if they exist
    if 'case_data' in glog.columns:
        # Parse case_data strings into dictionaries
        glog['case_data_parsed'] = glog['case_data'].apply(parse_dict_string)
        
        # Merge case_data dictionaries for each case_id
        case_data_merged = glog.groupby('case_id')['case_data_parsed'].agg(merge_dicts).reset_index()
        
        case_data_merged = case_data_merged.rename(columns={'case_data_parsed': 'case_data_dict'})

        
        # Map merged case data back to all rows for each case
        glog = glog.merge(case_data_merged, on='case_id', how='left')

        
        # Extract individual case attributes as separate columns
        case_attrs = set()
        for case_dict in glog['case_data_dict'].dropna():
            case_attrs.update(case_dict.keys())
        
        for attr in case_attrs:
            glog[f'case_{attr}'] = glog['case_data_dict'].apply(lambda d: d.get(attr) if isinstance(d, dict) else None)
    
    if 'event_data' in glog.columns:
        # Parse event_data strings into dictionaries
        glog['event_data_parsed'] = glog['event_data'].apply(parse_dict_string)
        
        # Extract individual event attributes as separate columns
        event_attrs = set()
        for event_dict in glog['event_data_parsed'].dropna():
            event_attrs.update(event_dict.keys())
        
        for attr in event_attrs:
            glog[f'event_{attr}'] = glog['event_data_parsed'].apply(lambda d: d.get(attr) if isinstance(d, dict) else None)
    
    # Filter to only completed processes
    glog = glog[glog['case_id'].isin(completed_process_ids)].copy()
    
    # Normalize known BPMN/PM4Py column variants
    if 'case:concept:name' in glog.columns and 'concept:name' in glog.columns:
        glog = glog.rename(columns={
            'case:concept:name': 'case_id',
            'concept:name': 'Activity',
            'time:timestamp': 'end_time',
            'org:resource': 'resource'
        })

    # Ensure required columns exist
    for col in ['case_id', 'Activity', 'end_time']:
        if col not in glog.columns:
            glog[col] = pd.NA

    # parse timestamps to UTC and drop invalid rows
    glog['end_time'] = pd.to_datetime(glog['end_time'], errors='coerce', utc=True, format='mixed')
    glog = glog.dropna(subset=['case_id', 'end_time'])

    # compute earliest timestamp per case and pick first N cases by start time
    case_start_times = glog.groupby('case_id')['end_time'].min()
    glog = glog.assign(case_start_time=glog['case_id'].map(case_start_times))
    ordered_cases = glog.sort_values(by=['case_start_time', 'case_id'])['case_id'].drop_duplicates().iloc[:max(0, int(num_traces))]
    selected_case_ids = ordered_cases.tolist()

    subset_glog = glog[glog['case_id'].isin(selected_case_ids)].copy()

    # normalize activity names and resource column
    subset_glog['Activity'] = subset_glog['Activity'].astype(str).str.lower()
    if 'resource' in subset_glog.columns:
        subset_glog = subset_glog.rename(columns={'resource': 'Resource'})
    else:
        subset_glog['Resource'] = pd.NA

    subset_glog = subset_glog.sort_values(by=['case_id', 'end_time']).reset_index(drop=True)
    print(len(subset_glog['case_id'].unique()), 'cases in the generated log')
    print(len(subset_glog), 'events in the generated log')
    # replace activity "srm: in transfer to execution system" with "srm: in transfer to execution system"
    subset_glog['Activity'] = subset_glog['Activity'].str.replace("srm: in transfer to execution system", "srm: in transfer to execution syst.", regex=False)
    return num_traces, subset_glog


def process_rlog(in_file, num_traces):
    # Try reading; support files with or without PM4Py-style headers
    try:
        rlog = pd.read_csv(in_file, header=0)
    except Exception:
        rlog = pd.read_csv(in_file, header=None)

    # Normalize known column names if present
    if 'case:concept:name' in rlog.columns and 'concept:name' in rlog.columns:
        rlog = rlog.rename(columns={
            'case:concept:name': 'case_id',
            'concept:name': 'Activity',
            'time:timestamp': 'end_time',
            'org:resource': 'Resource'
        })

    # If the file already uses 'case_id' / 'Activity' / 'end_time', keep them
    for col in ['case_id', 'Activity', 'end_time']:
        if col not in rlog.columns:
            rlog[col] = pd.NA

    # Parse timestamps to UTC and drop invalid rows
    rlog['end_time'] = pd.to_datetime(rlog['end_time'], errors='coerce', utc=True)
    rlog = rlog.dropna(subset=['case_id', 'end_time'])

    # Compute earliest timestamp per case and select first num_traces cases by that start time
    case_start_times = rlog.groupby('case_id')['end_time'].min()
    rlog = rlog.assign(case_start_time=rlog['case_id'].map(case_start_times))
    ordered_cases = rlog.sort_values(by=['case_start_time', 'case_id'])['case_id'].drop_duplicates().iloc[:max(0, int(num_traces))]
    selected_case_ids = ordered_cases.tolist()

    subset_rlog = rlog[rlog['case_id'].isin(selected_case_ids)].copy()

    # Ensure Resource column exists and normalize Activity
    if 'Resource' not in subset_rlog.columns and 'resource' in subset_rlog.columns:
        subset_rlog = subset_rlog.rename(columns={'resource': 'Resource'})
    if 'Resource' not in subset_rlog.columns:
        subset_rlog['Resource'] = pd.NA

    subset_rlog['Activity'] = subset_rlog['Activity'].astype(str).str.lower()

    subset_rlog = subset_rlog.sort_values(by=['case_id', 'end_time']).reset_index(drop=True)
    print(len(subset_rlog['case_id'].unique()), 'cases in the real log')
    print(len(subset_rlog), 'events in the real log')
    # remove all activities "srm: change was transmitted"
    subset_rlog = subset_rlog[~subset_rlog['Activity'].str.contains("srm: change was transmitted", na=False)]
    return subset_rlog

  

def main():
    # Preprocess generated log
    in_file_glog = './logs/p2p_3WayMatchingEC_zeroshot_tool_reflect_01_300traces_01.csv' #'./real_event_logs/BPIChallenge2019_3WayMatchingEC_cleaned.csv' #
    tiny_db_json_path = './db_zeroshot_tool_reflect_01_300traces_01/simulation_store.json'
    num_g_traces, gen_log = preprocess_glog(in_file=in_file_glog, tiny_db_json_path=tiny_db_json_path, num_traces=500)

    # Preprocess real log
    in_file_rlog = './real_event_logs/BPIChallenge2019_3WayMatchingEC_cleaned.csv' #_cleaned
    real_log = process_rlog(in_file=in_file_rlog, num_traces=num_g_traces)


    # Calculate metrics basing on the log-distance measures
    event_log_ids = EventLogIDs(
        case='case_id',
        activity='Activity',
        start_time='start_time',
        end_time='end_time',
        resource='Resource'
    )

    real_log[event_log_ids.end_time] = pd.to_datetime(real_log[event_log_ids.end_time], utc=True)
    real_log[event_log_ids.start_time] = real_log[event_log_ids.end_time] - timedelta(minutes=30)
    #real_log[event_log_ids.start_time] = pd.to_datetime(real_log[event_log_ids.end_time], utc=True)

   
    gen_log[event_log_ids.end_time] = pd.to_datetime(
        gen_log[event_log_ids.end_time],
        utc=True,
        format='mixed'  # to sovlve the mixed format issue
    )
    gen_log[event_log_ids.start_time] = pd.to_datetime(
        gen_log[event_log_ids.start_time],
        utc=True,
        format='mixed'  # to sovlve the mixed format issue
    )
    #gen_log[event_log_ids.start_time] = gen_log[event_log_ids.end_time] - timedelta(minutes=30)

    from log_distance_measures.config import DEFAULT_CSV_IDS
    from log_distance_measures.control_flow_log_distance import control_flow_log_distance

    distance = control_flow_log_distance(
        real_log, DEFAULT_CSV_IDS,
        gen_log, DEFAULT_CSV_IDS,
    )
    print(f'Control flow distance: {distance}')

    from log_distance_measures.n_gram_distribution import n_gram_distribution_distance

    distance = n_gram_distribution_distance(
        real_log, DEFAULT_CSV_IDS,
        gen_log, DEFAULT_CSV_IDS,
        n=3
    )
    print(f'3-gram distribution distance: {distance}')

    distance = n_gram_distribution_distance(
        real_log, DEFAULT_CSV_IDS,
        gen_log, DEFAULT_CSV_IDS,
        n=2
    )
    print(f'2-gram distribution distance: {distance}')

    from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
    from log_distance_measures.config import AbsoluteTimestampType, DEFAULT_CSV_IDS, discretize_to_hour

    # Call passing the event logs, its column ID mappings, timestamp type, and discretize function
    distance = absolute_event_distribution_distance(
        gen_log, DEFAULT_CSV_IDS,  # First event log and its column id mappings
        real_log, DEFAULT_CSV_IDS,  # Second event log and its column id mappings
        discretize_type=AbsoluteTimestampType.END,  # Which timestamps to consider (start times and/or end times)
        discretize_event=discretize_to_hour  # Function to discretize the time of each timestamp (default by hour)
    )
    print(f'Absolute event distribution distance: {distance}')

    from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
    from log_distance_measures.config import DEFAULT_CSV_IDS, discretize_to_hour

    distance = case_arrival_distribution_distance(
        gen_log, DEFAULT_CSV_IDS,  # First event log and its column id mappings
        real_log, DEFAULT_CSV_IDS,  # Second event log and its column id mappings
        discretize_event=discretize_to_hour  # Function to discretize each timestamp (default by hour)
    )
    print(f'Case arrival distribution distance: {distance}')

    from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
    from log_distance_measures.config import AbsoluteTimestampType, DEFAULT_CSV_IDS

    distance = circadian_event_distribution_distance(
        gen_log, DEFAULT_CSV_IDS,  # First event log and its column id mappings
        real_log, DEFAULT_CSV_IDS,  # Second event log and its column id mappings
        discretize_type=AbsoluteTimestampType.END  # Consider both start/end timestamps of each activity instance
    )
    print(f'Circadian event distribution distance: {distance}')

    from log_distance_measures.config import AbsoluteTimestampType, DEFAULT_CSV_IDS, discretize_to_hour
    from log_distance_measures.relative_event_distribution import relative_event_distribution_distance

    # Call passing the event logs, its column ID mappings, timestamp type, and discretize function
    distance = relative_event_distribution_distance(
        gen_log, DEFAULT_CSV_IDS,  # First event log and its column id mappings
        real_log, DEFAULT_CSV_IDS,  # Second event log and its column id mappings
        discretize_type=AbsoluteTimestampType.END,  # Which timestamps to consider (start times and/or end times)
        discretize_event=discretize_to_hour  # Function to discretize the time of each timestamp (default by hour)
    )
    print(f'Relative event distribution distance: {distance}')

    from log_distance_measures.config import DEFAULT_CSV_IDS
    from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance

    distance = cycle_time_distribution_distance(
        gen_log, DEFAULT_CSV_IDS,  # First event log and its column id mappings
        real_log, DEFAULT_CSV_IDS,  # Second event log and its column id mappings
        bin_size=pd.Timedelta(hours=1)  # Bins of 1 hour
    )
    print(f'Cycle time distribution distance: {distance}')

    from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
    from log_distance_measures.config import DEFAULT_CSV_IDS

    distance = circadian_workforce_distribution_distance(
        gen_log, DEFAULT_CSV_IDS,  # First event log and its column id mappings
        real_log, DEFAULT_CSV_IDS,  # Second event log and its column id mappings
    )
    print(f'Circadian workforce distribution distance: {distance}')

    # Generate statistics files about trace variants
    gen_stats_file = './logs/gen_stats.txt'
    real_stats_file = './logs/real_stats.txt'
    print(f'Generating trace variants statistics for generated log...')
    trace_variants(log_df=gen_log, case_id_col='case_id', activity_col='Activity', timestamp_col='end_time', out_file=gen_stats_file)
    print(f'Generating trace variants statistics for real log...')
    trace_variants(log_df=real_log, case_id_col='case_id', activity_col='Activity', timestamp_col='end_time', out_file=real_stats_file)
    
    check_characteristics(gen_log)

    atr_variability_gen = check_attribute_variability(
        log_df=gen_log,
        case_id_col='case_id',
        case_attrs=['case_spend_area', 'case_sub_spend_area', 'case_net_worth', 'case_vendor'],
        event_attrs=['event_task_difficulty', 'event_task_duration', 'event_satisfaction']
    )
    print('Attribute variability statistics for generated log:')
    pretty_print_attribute_variability(atr_variability_gen)

    #check_characteristics(real_log)
    atr_variability_real = check_attribute_variability(
        log_df=real_log,
        case_id_col='case_id',
        case_attrs=['Cumulative net worth (EUR)', 'case:Spend area text', 'case:Sub spend area text', 'case:Vendor'],
        event_attrs=[]
    )
    print('Attribute variability statistics for real log:')
    #pretty_print_attribute_variability(atr_variability_real)


    plot_string_distribution(
        df=real_log,
        attr='case:Vendor',
        top_n=20
    )
    plot_string_distribution(
        df=gen_log,
        attr='case_vendor',
        top_n=20
    )
    plot_string_distribution(
        df=real_log,
        attr='case:Spend area text',
        top_n=20
    )
    plot_string_distribution(
        df=gen_log,
        attr='case_spend_area',
        top_n=20
    )
    plot_string_distribution(
        df=real_log,
        attr='case:Sub spend area text',
        top_n=20
    )
    plot_string_distribution(
        df=gen_log,
        attr='case_sub_spend_area',
        top_n=20
    )
    plot_numeric_distributions(
        gen_df=gen_log,
        gen_attr='case_net_worth',
        real_df=real_log,
        real_attr='Cumulative net worth (EUR)',
        bins=30
    )


if __name__ == "__main__":
    main()