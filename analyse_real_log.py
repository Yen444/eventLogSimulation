import pandas as pd
from typing import List, Dict, Any
import pm4py
from pm4py.statistics.variants.log import get as variants_get
from collections import defaultdict
import re


def extract_log_stats(in_xes_file: str, out_txt_file: str, out_csv_file: str, case_id_col: str = "case:concept:name", activity_col: str = "concept:name", time_col: str = "time:timestamp", resource_col: str = "org:role"):
    log_df = pm4py.read_xes(in_xes_file)
    # save log in csv file
    if out_csv_file:
        log_df.to_csv(out_csv_file, index=False)
    
    log_df = clean_log_df(log_df, case_id_col, activity_col, time_col, resource_col)

    # get statistics and saved in a txt file
    # get num case and num activities and list of unique roles
    num_cases = log_df[case_id_col].nunique()
    num_activities = log_df[activity_col].nunique()
    unique_roles = log_df[resource_col].unique()

    # get all column names
    columns = log_df.columns.tolist()


    # get longest (case with most activities) and shortest (case with least activities) cases
    case_activity_counts = log_df[case_id_col].value_counts()
    longest_case = case_activity_counts.idxmax()
    shortest_case = case_activity_counts.idxmin()
    # get all rows of longest case
    longest_case_rows = log_df[log_df[case_id_col] == longest_case]
    # get all rows of shortest case
    shortest_case_rows = log_df[log_df[case_id_col] == shortest_case]
    

    with open(out_txt_file, 'w') as f:
        f.write(f"Columns in the log: {', '.join(columns)}\n\n")
        # extract the len of unique values and the first 10 unique values for the following columns: case:Project, case:OrganizationalEntity, case:Cost Type,case:RequestedAmount
        for col in ["case:Project", "case:OrganizationalEntity", "case:Cost Type", "case:RequestedAmount"]:
            unique_values = log_df[col].unique()
            f.write(f"Column '{col}': {len(unique_values)} unique values, first 10: {list(unique_values[:10])}\n")
        f.write("\n")
        f.write(f"Number of cases: {num_cases}\n")
        f.write(f"Number of activities: {num_activities}\n")
        f.write(f"Unique roles: {', '.join(unique_roles)}\n\n")
        f.write(f"Longest case ID: {longest_case} with {case_activity_counts.max()} activities\n")
        f.write(f"Shortest case ID: {shortest_case} with {case_activity_counts.min()} activities\n")

    # Write all rows of the longest case (sorted by timestamp) to the txt file — placed near the top
    try:
        # correctly select specific columns for the longest case using .loc and a list of column names
        longest_case_rows_sorted = (
            log_df.loc[log_df[case_id_col] == longest_case, [activity_col, time_col, resource_col]]
            .sort_values(by=time_col, ascending=True)
        )

        with open(out_txt_file, 'a') as f:
            # write header without a preceding blank line
            f.write("Events for the longest case (sorted by time):\n")
            if longest_case_rows_sorted.empty:
                f.write("(no events found for the longest case)\n")
            else:
                # generate CSV text and normalize line endings to avoid extra blank lines
                csv_text = longest_case_rows_sorted.to_csv(index=False)
                # normalize any CRLF or CR to LF, then collapse multiple blank lines
                csv_text = csv_text.replace('\r\n', '\n').replace('\r', '\n')
                csv_text = re.sub(r'\n{2,}', '\n', csv_text)
                if not csv_text.endswith('\n'):
                    csv_text += '\n'
                f.write(csv_text)
    except Exception as e:
        with open(out_txt_file, 'a') as f:
            f.write(f"\nCould not write longest case rows due to error: {e}\n")

    # map activities to resources and their frequency
    activity_resource_counts = log_df.groupby(activity_col)[resource_col].value_counts()
    # Restructure into: {activity: [(resource_1, count), (resource_2, count), ...]}
    activity_resources = defaultdict(list)
    for (activity, resource), count in activity_resource_counts.items():
        activity_resources[activity].append((resource, count))  
    # Optional: convert to plain dict for display or export
    activity_resources = dict(activity_resources)
    # write activity resources to the file
    with open(out_txt_file, 'a') as f:
        f.write("\nActivities and their resources with frequency:\n")
        for activity, resources in activity_resources.items():
            f.write(f"{activity}: {resources}\n")
    
    # mine arrival time

    # Extract first event timestamp per case (case arrival) 
    case_arrivals = (
        log_df.groupby(case_id_col)['time:timestamp']
        .min()
        .reset_index(name='case_arrival')
    )

    # Extract time features
    case_arrivals['arrival_hour'] = case_arrivals['case_arrival'].dt.hour
    case_arrivals['arrival_dayofweek'] = case_arrivals['case_arrival'].dt.dayofweek
    case_arrivals['arrival_month'] = case_arrivals['case_arrival'].dt.month

    # Compute arrival counts 
    arrival_rate_hour = case_arrivals.groupby('arrival_hour').size()
    arrival_rate_dayofweek = case_arrivals.groupby('arrival_dayofweek').size()
    arrival_rate_month = case_arrivals.groupby('arrival_month').size()

    # Compute average arrivals per time unit
    avg_per_hour = arrival_rate_hour / case_arrivals['case_arrival'].dt.date.nunique()
    avg_per_day = arrival_rate_dayofweek / case_arrivals['case_arrival'].dt.to_period('W').nunique()
    avg_per_month = arrival_rate_month / case_arrivals['case_arrival'].dt.to_period('M').nunique()

    # Write results to file 
    def write_series(f, title, series):
        f.write(f"{title}:\n{series.to_string()}\n\n")

    with open(out_txt_file, 'a') as f:
        write_series(f, "Arrival rate per hour of day", arrival_rate_hour)
        write_series(f, "Arrival rate per day of week (0=Mon ... 6=Sun)", arrival_rate_dayofweek)
        write_series(f, "Arrival rate per month (1=Jan ... 12=Dec)", arrival_rate_month)
        write_series(f, "Average cases arriving per hour", avg_per_hour)
        write_series(f, "Average cases arriving per day", avg_per_day)
        write_series(f, "Average cases arriving per month", avg_per_month)

    # Calculate inter-arrival times
    if len(case_arrivals) > 1:
        case_arrivals_sorted = case_arrivals.sort_values('case_arrival')
        inter_arrival_hours = case_arrivals_sorted['case_arrival'].diff().dt.total_seconds() / 3600
        inter_arrival_hours = inter_arrival_hours.dropna()
        
        mean_inter_arrival = inter_arrival_hours.mean()
        std_inter_arrival = inter_arrival_hours.std()
        
        with open(out_txt_file, 'a') as f:
            f.write(f"Mean inter-arrival time: {mean_inter_arrival:.2f} hours\n")
            f.write(f"Standard deviation of inter-arrival time: {std_inter_arrival:.2f} hours\n")


    # Base average hourly arrival rate
    base_avg_hourly_rate = avg_per_hour.sum() / 24
    with open(out_txt_file, 'a') as f:
        f.write(f"Base average hourly arrival rate: {base_avg_hourly_rate:.2f} cases/hour\n")

    # Print total number of cases
    print(f"Total number of cases: {len(case_arrivals)}")

    # Trace variants
    # get trace variants, sort by frequency and write to this out_text_file
    trace_variants = get_trace_variants(log_df, case_id_col, activity_col, time_col)
    # print number of variants
    print(len(trace_variants), 'variants found')
    # print average len of variants
    avg_len = sum(len(v) for v in trace_variants.keys()) / len(trace_variants.keys()) if trace_variants else 0
    print(f"Average length of variants: {avg_len:.2f}")
    # sort variants by their frequency (number of traces) descending
    sorted_variants = sorted(trace_variants.items(), key=lambda item: len(item[1]), reverse=True)
    # write all variants ordered by frequency (descending)
    with open(out_txt_file, 'a', encoding='utf-8') as f:
        f.write(f'{len(trace_variants)} variants found. Average length: {avg_len:.2f}\n\n')
        for idx, (variant, traces) in enumerate(sorted_variants, start=1):
            freq = len(traces)
            f.write(f"Variant #{idx}: {variant}\n")
            f.write(f"Frequency: {freq}\n\n")
            # List trace belonging to this variant
            #for trace in traces:
            #    f.write(f"  Trace: {trace}\n")
            

def get_trace_variants(log_df, case_id_col, activity_col, timestamp_col):
    """Get trace variants from the event log.
    Parameters:
        in_file: path the input CSV file.

    Returns:
        trace variants
    """

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

    return variants

def discover_bpmn_model(log_xes_file, case_id_col="case:concept:name", activity_col="concept:name", timestamp_col="time:timestamp", resource_col="org:role", out_bpmn_file=None, noise_threshold=0.3):
    # Load the event log
    log_df = pm4py.read_xes(log_xes_file)
    log_df = clean_log_df(log_df, case_id_col, activity_col, timestamp_col, resource_col)

    # Discover the BPMN model

    process_model = pm4py.discover_bpmn_inductive(
        log_df,
        noise_threshold=noise_threshold,
        activity_key=activity_col,
        case_id_key=case_id_col,
        timestamp_key=timestamp_col)
    if out_bpmn_file:
        pm4py.write_bpmn(process_model, out_bpmn_file, auto_layout=True)

def clean_log_df(log_df, case_id_col: str = "case:concept:name", activity_col: str = "concept:name", timestamp_col: str = "time:timestamp", resource_col: str = "org:role"):
    # drop rows with missing case id, activitiy name, timestamp, or resource
    log_df = log_df.dropna(subset=[case_id_col, activity_col, timestamp_col, resource_col])
    # convert timestamp to datetime
    #log_df[timestamp_col] = pd.to_datetime(
    #    log_df[timestamp_col],
    #    utc=True,
    #    format='mixed'  # to sovlve the mixed format issue
    #)
    # lower activity names
    log_df[activity_col] = log_df[activity_col].str.lower()
    # convert activities names basing on given mapping
    activity_mapping = {}
    log_df[activity_col] = log_df[activity_col].replace(activity_mapping)

    # remove traces that have less than 2 activities
    activity_counts = log_df.groupby(case_id_col)[activity_col].transform('count')
    log_df = log_df[activity_counts >= 2]

    # remove traces that have any activities done by MISSING role
    invalid_cases = log_df[log_df[resource_col].str.upper() == "MISSING"][case_id_col].unique()
    log_df = log_df[~log_df[case_id_col].isin(invalid_cases)]

    # print all activities for case "request for payment 148749" and sorted by timestamp ascending
    #case_id = "request for payment 148157"
    #case_activities = log_df[log_df[case_id_col] == case_id].sort_values(by=timestamp_col)

    return log_df


def main():
    log_xes_file = "./real_event_logs/RequestForPayment.xes"
    out_csv_file = "./real_event_logs/RequestForPayment.csv"
    out_txt_file = "./real_event_logs/logStats_BPI2020_RequestForPayment.txt"
    extract_log_stats(log_xes_file, out_txt_file, out_csv_file)
    out_bpmn_file = "./bpmn_files/RequestForPayment.bpmn"
    discover_bpmn_model(log_xes_file, out_bpmn_file=out_bpmn_file, noise_threshold=0.3)

if __name__ == "__main__":
    main()