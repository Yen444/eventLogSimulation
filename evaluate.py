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

def check_characteristics(event_log: pd.DataFrame):
    """
    Characteristic 1: "Female employees take more time to complete tasks than male employees."
    Visualize with a boxplot using matplotlib only.
    """

    # Convert timestamps
    event_log['start_timestamp'] = pd.to_datetime(event_log['start_time'])
    event_log['end_timestamp'] = pd.to_datetime(event_log['end_time'])

    # Compute duration in seconds
    event_log['duration'] = (event_log['end_timestamp'] - event_log['start_timestamp']).dt.total_seconds()

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
    plt.show()

    # Print mean durations for reference
    mean_duration = df.groupby('gender')['duration'].mean()
    print("Mean duration by gender:\n", mean_duration)


    # --- Characteristic 2: Senior employees for high networth SRM ---
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
                #for trace in traces:
                #    f.write(f"  Trace: {trace}\n")
                f.write(f"  Frequency: {freq}\n\n")

def preprocess_glog(in_file, num_traces=692):
    """Preprocess a generated log CSV into a normalized DataFrame.

    Steps:
    - Try reading with header; if header missing expected columns, fall back to headerless read with defaults
    - Normalize column names to 'case_id', 'Activity', 'end_time', 'Resource'
    - Parse timestamps to UTC and drop invalid rows
    - Select the first `num_traces` cases ordered by case start time
    - Lowercase activity names and sort events by case and timestamp

    Returns a pandas.DataFrame with columns at least: 'case_id', 'Activity', 'end_time', 'Resource'.
    """
    default_col_names = ['iteration_id', 'case_id', 'Activity', 'agent_id', 'start_time', 'end_time', 'role', 'resource', 'case_data', 'event_data', 'message', 'end', 'process_state']
    # try reading with header, fall back to headerless
    try:
        glog = pd.read_csv(in_file, header=0)
        if not any(c in glog.columns for c in ['case_id', 'case:concept:name']):
            raise ValueError('header missing expected case id')
    except Exception:
        glog = pd.read_csv(in_file, header=None, names=default_col_names)
    
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
    # replace activity "srm: in transfer to execution system" with "srm: in transfer to execution system"
    subset_glog['Activity'] = subset_glog['Activity'].str.replace("srm: in transfer to execution system", "srm: in transfer to execution syst.", regex=False)
    return subset_glog


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
    # remove all activities "srm: change was transmitted"
    subset_rlog = subset_rlog[~subset_rlog['Activity'].str.contains("srm: change was transmitted", na=False)]
    return subset_rlog

   

def main():
    # Preprocess generated log
    in_file_glog = './logs/p2p_3WayMatchingEC_01.csv' #'./real_event_logs/BPIChallenge2019_3WayMatchingEC_cleaned.csv' #
    gen_log = preprocess_glog(in_file=in_file_glog, num_traces=500)

    # Preprocess real log
    in_file_rlog = './real_event_logs/BPIChallenge2019_3WayMatchingEC.csv' #_cleaned
    real_log = process_rlog(in_file=in_file_rlog, num_traces=500)


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
    
    #check_characteristics(gen_log)

if __name__ == "__main__":
    main()