import pandas as pd
import math
from log_distance_measures.config import EventLogIDs
import numpy as np
from datetime import timedelta
import pm4py
import matplotlib.pyplot as plt
import re
import ast
from typing import List, Dict, Any

# Import utility functions
from evaluate_utils import (
    filter_complete_processes,
    parse_dict_string,
    merge_dicts,
    extract_gender,
    plot_numeric_distributions,
    plot_numeric_distribution,
    plot_string_distributions,
    plot_string_distribution,
    generate_process_model,
    trace_variants,
    visualize_logs_2d
)

# Import metric calculation functions
from metrics import (
    calculate_all_metrics,
    compare_logs_with_baseline,
    prepare_log_for_metrics
)

def check_characteristics(event_log: pd.DataFrame, outfile=None, figsize=(6, 4)):
    """
    Characteristic 1_0: "Female employees take more time to complete tasks than male employees."
    Characteristic 1_1: "Male employees take more time to complete tasks than female employees."
    Visualize with a boxplot using matplotlib only.
    """

    # Convert timestamps
    event_log['start_timestamp'] = pd.to_datetime(event_log['start_time'])
    event_log['end_timestamp'] = pd.to_datetime(event_log['end_time'])

    # Compute duration in seconds
    # duration is in task_duration inside event_data column, in hours (string)
    event_log['duration'] = event_log['event_data'].str.extract(r'task_duration:\s*(\d+\.?\d*)').astype(float) * 3600  # convert hours to seconds
    # Extract gender from agent_data (example: "name: Klaas Jansen, age: 42, gender: Male")
    event_log['gender'] = event_log['Resource'].apply(extract_gender)

    # Drop rows without duration or gender
    df = event_log.dropna(subset=['duration', 'gender'])

    # Prepare data for boxplot
    male_durations = df[df['gender'] == 'Male']['duration']
    female_durations = df[df['gender'] == 'Female']['duration']

    # Calculate mean and median durations for display
    mean_duration = df.groupby('gender')['duration'].mean()
    median_duration = df.groupby('gender')['duration'].median()
    
    # Boxplot
    plt.figure(figsize=figsize)
    plt.boxplot([male_durations, female_durations], tick_labels=['Male', 'Female'])
    plt.title('Task Duration by Gender')
    plt.ylabel('Duration (seconds)')
    plt.xlabel('Gender')
    plt.grid(True, alpha=0.3)
    
    # Add mean and median duration text annotations on the plot
    male_mean = mean_duration.get('Male', 0)
    female_mean = mean_duration.get('Female', 0)
    male_median = median_duration.get('Male', 0)
    female_median = median_duration.get('Female', 0)
    
    offset = 500  # adjust this based on your y-scale

    if male_mean > 0:
        plt.text(0.6, male_mean + 3*offset, f'Mean:{male_mean:.0f}s', 
                ha='left', va='bottom', fontweight='bold', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))

    if male_median > 0:
        plt.text(0.6, male_median - 3*offset, f'Median:{male_median:.0f}s', 
                ha='left', va='top', fontweight='bold', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))

    if female_mean > 0:
        plt.text(2.2, female_mean + 3*offset, f'Mean:{female_mean:.0f}s', 
                ha='left', va='bottom', fontweight='bold', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightpink', alpha=0.3))

    if female_median > 0:
        plt.text(2.2, female_median - 3*offset, f'Median:{female_median:.0f}s', 
                ha='left', va='top', fontweight='bold', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightpink', alpha=0.3))
    
    
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

    # Print mean and median durations for reference
    print("Mean duration by gender:\n", mean_duration)
    print("Median duration by gender:\n", median_duration)


    # --- Characteristic 2: Senior employees for high networth SRM ---

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
    print(len(completed_process_ids), 'completed processes ID found in the simulation store')
    if len(completed_process_ids) < num_traces:
        num_traces = len(completed_process_ids)
    
    default_col_names = ['iteration_id', 'case_id', 'Activity', 'agent_id', 'start_time', 'end_time', 'role', 'resource', 'case_data', 'event_data', 'message', 'reasoning', 'end', 'process_state']
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
    # replace activity "srm: in transfer to execution system" with "srm: in transfer to execution syst."
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
    #in_file_glog = './logs/p2p_3WayMatchingEC_zeroshot_tool_reflect_01_300traces_openai.csv' #'./real_event_logs/BPIChallenge2019_3WayMatchingEC_cleaned.csv' #
    #tiny_db_json_path = './db_zeroshot_tool_reflect_01_300traces_openai/simulation_store.json'
    print('Preprocessing our generated log...')
    in_file_glog = './logs/test_2.csv'
    tiny_db_json_path = './db_2/simulation_store.json'
    num_g_traces, our_gen_log = preprocess_glog(in_file=in_file_glog, tiny_db_json_path=tiny_db_json_path, num_traces=464) #464

    # Preprocess real log
    print('Preprocessing real log...')
    in_file_rlog = './real_event_logs/BPIChallenge2019_3WayMatchingEC.csv' #_cleaned
    real_log = process_rlog(in_file=in_file_rlog, num_traces=num_g_traces)

    # plg2 log
    print('Preprocessing plg2 log...')
    in_file_plg2log = './logs/p2p_plg2.csv'
    plg2_log = process_rlog(in_file=in_file_plg2log, num_traces=num_g_traces)
    print(f'PLG2 columns: {plg2_log.columns.tolist()}')
    # print unique columns of resource
    print(f'PLG2 unique resources: {plg2_log["Resource"].unique()}')

    # Calculate metrics for our generated log
    print('Metrics for our generated log:')
    results = calculate_all_metrics(log1=real_log, log2=our_gen_log)
   
    # Calculate metrics for PLG2 log
    print('Metrics for PLG2 generated log:')
    plg2_results = calculate_all_metrics(log1=real_log, log2=plg2_log)
    
    # Generate statistics files about trace variants
    gen_stats_file = './logs/gen_stats.txt'
    real_stats_file = './logs/real_stats.txt'
    print(f'Generating trace variants statistics for generated log...')
    generated_trace_variants = trace_variants(log_df=our_gen_log, case_id_col='case_id', activity_col='Activity', timestamp_col='end_time', out_file=gen_stats_file)
    print(f'Generating trace variants statistics for real log...')
    real_trace_variants = trace_variants(log_df=real_log, case_id_col='case_id', activity_col='Activity', timestamp_col='end_time', out_file=real_stats_file)
    print('Generating trace variants for PLG2 log...')
    plg2_trace_variants = trace_variants(log_df=plg2_log, case_id_col='case_id', activity_col='Activity', timestamp_col='end_time', out_file='./logs/plg2_stats.txt')
    
    check_characteristics(our_gen_log, outfile='images/gen_log_characteristics.png', figsize=(8, 9))

    plot_string_distribution(
        df=real_log,
        attr='case:Vendor',
        top_n=20,
        title='Distribution of Real Vendor',
        outfile='images/real_vendor_distribution.png',
        figsize=(12, 8)
    )
    plot_string_distribution(
        df=our_gen_log,
        attr='case_vendor',
        top_n=20,
        title='Distribution of Generated Vendor',
        outfile='images/gen_vendor_distribution.png',
        color='red',
        figsize=(12, 8)
    )
    plot_string_distribution(
        df=real_log,
        attr='case:Spend area text',
        top_n=20,
        title='Distribution of Real Spend Area',
        outfile='images/real_spend_area_distribution.png',
        figsize=(12, 8)
    )
    plot_string_distribution(
        df=our_gen_log,
        attr='case_spend_area',
        top_n=20,
        title='Distribution of Generated Spend Area',
        outfile='images/gen_spend_area_distribution.png',
        color='red',
        figsize=(12, 8)
    )
    plot_string_distribution(
        df=real_log,
        attr='case:Sub spend area text',
        top_n=20,
        title='Distribution of Real Sub Spend Area',
        outfile='images/real_sub_spend_area_distribution.png',
        figsize=(12, 8)
    )
    plot_string_distribution(
        df=our_gen_log,
        attr='case_sub_spend_area',
        top_n=20,
        title='Distribution of Generated Sub Spend Area',
        outfile='images/gen_sub_spend_area_distribution.png',
        color='red',
        figsize=(12, 8)
    )
    plot_numeric_distributions(
        gen_df=our_gen_log,
        gen_attr='case_net_worth',
        real_df=real_log,
        real_attr='Cumulative net worth (EUR)',
        bins=30,
        title='Net Worth Distribution Comparison',
        outfile='images/net_worth_distribution.png',
        figsize=(10, 6)
    )
    plot_numeric_distribution(
        df=our_gen_log,
        attr='case_net_worth',
        bins=30,
        title='Distribution of Generated Net Worth',
        outfile='images/gen_net_worth_distribution.png',
        color='green',
        figsize=(10, 6)
    )
    log_variants = {
        'Real log': real_trace_variants,
        'Our generated log': generated_trace_variants,
        'PLG2 generated log': plg2_trace_variants
    }
    visualize_logs_2d(
        log_variants,
        method='pca',
        perplexity=20,
        outfile='images/logs_2d_visualization_pca_uncleaned_real_log.png',
        title='2D Visualization of Different Logs (PCA dimensionality reduction)'
    )
    # Trace variant distribution distance
    from evaluate_utils import calculate_trace_variant_distribution_distance
    distance_our_gen = calculate_trace_variant_distribution_distance(
        log1_variants_dict=generated_trace_variants,
        log2_variants_dict=real_trace_variants
    )
    print(f'Trace variant distribution distance (Our generated vs Real): {distance_our_gen}')
    distance_plg2 = calculate_trace_variant_distribution_distance(
        log1_variants_dict=plg2_trace_variants,
        log2_variants_dict=real_trace_variants
    )
    print(f'Trace variant distribution distance (PLG2 vs Real): {distance_plg2}')

if __name__ == "__main__":
    main()