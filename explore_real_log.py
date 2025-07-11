import pm4py
import pandas as pd
from collections import defaultdict
#log_file = './real_event_logs/BPIChallenge2019.xes'
#log = pm4py.read_xes(log_file) 

# filter 3 way matching before GR and save the filtered log as csv file
#log = log[(log["case:GR-Based Inv. Verif."] == False) & (log["case:Goods Receipt"] == True) & (log["case:Item Type"] != "Consignment")]
#log = log[(log["case:GR-Based Inv. Verif."] == False) & (log["case:Goods Receipt"] == True) & (log["case:Item Type"] != "Consignment") & (log["case:Document Type"] == "EC Purchase order")]
#log.to_csv('./real_event_logs/BPIChallenge2019_3WayMatchingEC.csv', index=False)

# filter 2 way matching before GR and save the filtered log as csv file
#log = log[log["case:Goods Receipt"] == False]
#log.to_csv('./real_event_logs/BPIChallenge2019_2WayMatching.csv', index=False)

def clean_log(in_log_path, out_log_path=None):
    """Function to filter log.
        Why do we have to filter log? 
            - To easily discover the parameters used for simulation
            - To use this filtered real log to evaluate the generated log
        How to filter? We need to remove traces, which:
            - Contain certain activities that are not included in the process model.
            - contain event with a missing resource (i.e., the resource is None)? May be only remove these activity.
            - contain contain resources whose most frequent activity occurs fewer than 20 times.
        Args:
            - path to input log csv file
            - path to output log
        Returns:
            - cleaned log
    """

    # read log file
    in_log = pd.read_csv(in_log_path)
    # column names
    case_col = "case:concept:name"
    activity_col = "concept:name"
    resource_col = "org:resource"

    ### keep traces with activity change price
    change_price_cases = in_log[in_log[activity_col] == 'Change Price'][case_col]
    change_price_cases_log = in_log[in_log[case_col].isin(change_price_cases)]

    ### Remove traces where all resources have low frequency (< 20) ---
    # count frequency of activities per resource
    resource_activity_freq = (
        in_log.groupby([resource_col, activity_col])
        .size()
        .reset_index(name="count")
    )
    # get max activity count per resource
    max_activity_count_per_resource = resource_activity_freq.groupby(resource_col)["count"].max()
    # identify low-frequency resources
    low_freq_resources = max_activity_count_per_resource[max_activity_count_per_resource < 20].index
    # add one special user
    low_freq_resources = low_freq_resources.union(pd.Index(['user_004']))
    # find traces where all resources are low-frequency
    cases_with_only_low_freq_resources = (
        in_log[in_log[resource_col].isin(low_freq_resources)][case_col].unique()
    )
    # remove those traces
    filtered_log = in_log[~in_log[case_col].isin(cases_with_only_low_freq_resources)]

    # Combine with change_price case
    #filtered_log = pd.concat([filtered_log, change_price_cases_log], ignore_index=True).drop_duplicates()
    
    #Remove only events with missing resources (not the whole trace) ---
    filtered_log = filtered_log[filtered_log[resource_col]!='NONE']

    ### remove certain events
    exclude_activities = ["Cancel Goods Receipt", "Cancel Invoice Receipt", "Cancel Subsequent Invoice", "Change Delivery Indicator", "Change Final Invoice Indicator", "Change Quantity", "Delete Purchase Order Item", "Record Subsequent Invoice", "SRM: Deleted", "SRM: Held", "SRM: Incomplete", "SRM: Transaction Completed"]
    filtered_log = filtered_log[~filtered_log[activity_col].isin(exclude_activities)]

    if out_log_path:
        filtered_log.to_csv(out_log_path)
    return filtered_log

    


def explore_log(log:pd.DataFrame, case_id_col="case:concept:name", resource_col="org:resource", activity_col="concept:name", output_file='./real_event_logs/logStats_BPI2019_3WayMatchingEC_cleaned.txt'):
    """
    Explore the event log and print some statistics.
    - Number of cases
    - Number of activities
    - Resource and their working periods
    - Activities and their resources along with frequencies
    - Arrival rate
    Args:
        log (DataFrame): The event log to explore.
        case_id_col (str): The column name for case IDs.
        resource_col (str): The column name for resources.
        activity_col (str): The column name for activities.
    """
    # Number of cases
    num_cases = log[case_id_col].nunique()

    # Number of unique activities
    num_activities = log[activity_col].nunique()


    # I want to map resource to the first timespan of the activity they performed and the last timespan of the activity they performed
    log_sorted = log.sort_values(by='time:timestamp')
    resource_first_last = log_sorted.groupby(resource_col).agg(
        first_timestamp=('time:timestamp', 'first'),
        last_timestamp=('time:timestamp', 'last')
    ).reset_index()

    # write information to a file
    with open(output_file, 'w') as f:
        f.write(f"Number of cases: {num_cases}\n")
        f.write(f"Number of unique activities: {num_activities}\n")
        f.write("Resources and their activities with frequency:\n")
        f.write("\nFirst and last timestamp for each resource:\n")
        for _, row in resource_first_last.iterrows():
            f.write(f"{row[resource_col]}: First Timestamp: {row['first_timestamp']}, Last Timestamp: {row['last_timestamp']}\n")
    
            
    # map activities to resources and their frequency
    activity_resource_counts = log.groupby(activity_col)[resource_col].value_counts()
    # Restructure into: {activity: [(resource_1, count), (resource_2, count), ...]}
    activity_resources = defaultdict(list)
    for (activity, resource), count in activity_resource_counts.items():
        activity_resources[activity].append((resource, count))  
    # Optional: convert to plain dict for display or export
    activity_resources = dict(activity_resources)
    # write activity resources to the file
    with open(output_file, 'a') as f:
        f.write("\nActivities and their resources with frequency:\n")
        for activity, resources in activity_resources.items():
            f.write(f"{activity}: {resources}\n")

    
    # Arrival time

    # Extract the first event timestamp per case and print also the concept:name of the first event
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    case_arrivals = log.groupby(case_id_col)['time:timestamp'].min().reset_index(name='case_arrival')
  
    # Extract time features
    case_arrivals['arrival_hour'] = case_arrivals['case_arrival'].dt.hour           # 0–23
    case_arrivals['arrival_dayofweek'] = case_arrivals['case_arrival'].dt.dayofweek # 0=Monday, 6=Sunday
    case_arrivals['arrival_month'] = case_arrivals['case_arrival'].dt.month         # 1=Jan, ..., 12=Dec

    #print('Number of cases:', len(case_arrivals))

    # Count how many cases arrived in each hour of the day
    arrival_rate_hour = case_arrivals.groupby('arrival_hour').size()

    #print(case_arrivals.head(10))  # Print first 10 case IDs for verification
    #print('Arrival rate per hour of day:')
    #print(arrival_rate_hour)
    import numpy as np
    #print(np.sum(arrival_rate_hour))  # Total number of cases

    # Count how many cases arrived on each day of the week
    arrival_rate_dayofweek = case_arrivals.groupby('arrival_dayofweek').size()

    #print('Arrival rate per day of week (0=Mon ... 6=Sun):')
    #print(arrival_rate_dayofweek)
    #print(np.sum(arrival_rate_dayofweek))  # Total number of cases

    # Count how many cases arrived in each month
    arrival_rate_month = case_arrivals.groupby('arrival_month').size()

    #print('Arrival rate per month (1=Jan ... 12=Dec):')
    #print(arrival_rate_month)
    #print(np.sum(arrival_rate_month))  # Total number of cases

    # (Optional) normalize to get average per day/hour/month
    # For example, average cases arriving *per hour* (assuming full 24-hour coverage):
    avg_per_hour = arrival_rate_hour / case_arrivals['case_arrival'].dt.date.nunique()
    avg_per_day = arrival_rate_dayofweek / (case_arrivals['case_arrival'].dt.to_period('W').nunique())
    avg_per_month = arrival_rate_month / (case_arrivals['case_arrival'].dt.to_period('M').nunique())

    # Output results (raw arrival rate, not normalized)
    with open(output_file, 'a') as f:
        f.write("Arrival rate per hour of day:\n")
        f.write(f"{arrival_rate_hour.to_string()}\n\n")
        
        f.write("Arrival rate per day of week (0=Mon ... 6=Sun):\n")
        f.write(f"{arrival_rate_dayofweek.to_string()}\n\n")
        
        f.write("Arrival rate per month (1=Jan ... 12=Dec):\n")
        f.write(f"{arrival_rate_month.to_string()}\n\n")

        f.write("Average cases arriving per hour:\n")
        f.write(f"{avg_per_hour.to_string()}\n\n")
        f.write("Average cases arriving per day:\n")
        f.write(f"{avg_per_day.to_string()}\n\n")
        f.write("Average cases arriving per month:\n")
        f.write(f"{avg_per_month.to_string()}\n\n")
    
    # calculate mean inter-arrival time
    if len(case_arrivals) > 1:
        # Sort by arrival time
        case_arrivals = case_arrivals.sort_values(by='case_arrival')
        # Calculate inter-arrival times in hours
        inter_arrival_times = (case_arrivals['case_arrival'].diff().dt.total_seconds() / 3600).dropna()
        mean_inter_arrival = inter_arrival_times.mean()
        std_inter_arrival = inter_arrival_times.std()
        
        with open(output_file, 'a') as f:
            f.write(f"Mean inter-arrival time: {mean_inter_arrival:.2f} hours\n")
            f.write(f"Standard deviation of inter-arrival time: {std_inter_arrival:.2f} hours\n")

    # calculate base average for hourly arrival rate = total of average arrivals per hour / 24
    total_avg_hourly_arrivals = avg_per_hour.sum()
    base_avg_hourly_arrival_rate = total_avg_hourly_arrivals / 24
    with open(output_file, 'a') as f:
        f.write(f"Base average hourly arrival rate: {base_avg_hourly_arrival_rate:.2f} cases/hour\n")
    
    # calculate arrival rate per hour, day of week, and month

    # Assuming log is your event log DataFrame
    # and 'case:concept:name' is your case ID column
    case_id_col = 'case:concept:name'

    # Extract the first event timestamp per case and print also the concept:name of the first event
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    case_arrivals = log.groupby(case_id_col)['time:timestamp'].min().reset_index(name='case_arrival')
  
    # Extract time features
    case_arrivals['arrival_hour'] = case_arrivals['case_arrival'].dt.hour           # 0–23
    case_arrivals['arrival_dayofweek'] = case_arrivals['case_arrival'].dt.dayofweek # 0=Monday, 6=Sunday
    case_arrivals['arrival_month'] = case_arrivals['case_arrival'].dt.month         # 1=Jan, ..., 12=Dec

    # Count how many cases arrived in each hour of the day
    arrival_rate_hour = case_arrivals.groupby('arrival_hour').size()

    # Count how many cases arrived on each day of the week
    arrival_rate_dayofweek = case_arrivals.groupby('arrival_dayofweek').size()

    # Count how many cases arrived in each month
    arrival_rate_month = case_arrivals.groupby('arrival_month').size()

    # (Optional) normalize to get average per day/hour/month
    # For example, average cases arriving *per hour* (assuming full 24-hour coverage):
    avg_per_hour = arrival_rate_hour / case_arrivals['case_arrival'].dt.date.nunique()
    avg_per_day = arrival_rate_dayofweek / (case_arrivals['case_arrival'].dt.to_period('W').nunique())
    avg_per_month = arrival_rate_month / (case_arrivals['case_arrival'].dt.to_period('M').nunique())

    # Output results (raw arrival rate, not normalized)
    with open(output_file, 'a') as f:
        f.write("Arrival rate per hour of day:\n")
        f.write(f"{arrival_rate_hour.to_string()}\n\n")
        
        f.write("Arrival rate per day of week (0=Mon ... 6=Sun):\n")
        f.write(f"{arrival_rate_dayofweek.to_string()}\n\n")
        
        f.write("Arrival rate per month (1=Jan ... 12=Dec):\n")
        f.write(f"{arrival_rate_month.to_string()}\n\n")

        f.write("Average cases arriving per hour:\n")
        f.write(f"{avg_per_hour.to_string()}\n\n")
        f.write("Average cases arriving per day:\n")
        f.write(f"{avg_per_day.to_string()}\n\n")
        f.write("Average cases arriving per month:\n")
        f.write(f"{avg_per_month.to_string()}\n\n")
    
    # calculate mean inter-arrival time
    if len(case_arrivals) > 1:
        # Sort by arrival time
        case_arrivals = case_arrivals.sort_values(by='case_arrival')
        # Calculate inter-arrival times in hours
        inter_arrival_times = (case_arrivals['case_arrival'].diff().dt.total_seconds() / 3600).dropna()
        mean_inter_arrival = inter_arrival_times.mean()
        std_inter_arrival = inter_arrival_times.std()
        
        with open(output_file, 'a') as f:
            f.write(f"Mean inter-arrival time: {mean_inter_arrival:.2f} hours\n")
            f.write(f"Standard deviation of inter-arrival time: {std_inter_arrival:.2f} hours\n")

    # calculate base average for hourly arrival rate = total of average arrivals per hour / 24
    total_avg_hourly_arrivals = avg_per_hour.sum()
    base_avg_hourly_arrival_rate = total_avg_hourly_arrivals / 24
    with open(output_file, 'a') as f:
        f.write(f"Base average hourly arrival rate: {base_avg_hourly_arrival_rate:.2f} cases/hour\n")




def main():
    in_log_file = './real_event_logs/BPIChallenge2019_3WayMatchingEC.csv'
    out_log_file = './real_event_logs/BPIChallenge2019_3WayMatchingEC_cleaned.csv'
    cleaned_log = clean_log(in_log_path=in_log_file, out_log_path=out_log_file)

    explore_log(cleaned_log)

    
    process_model = pm4py.discover_bpmn_inductive(
        cleaned_log,
        noise_threshold=0.3,
        activity_key='concept:name',
        case_id_key='case:concept:name',
        timestamp_key='time:timestamp')

    pm4py.write_bpmn(process_model, './bpmn_files/BPIChallenge2019_3WayMatchingEC_cleaned.bpmn', auto_layout=True)
if __name__ == "__main__":
    main()