import pandas as pd
from typing import List, Dict, Any
import json

def parse_data(string):
    # Ignore if string is None or 'null' or not a string
    if not isinstance(string, str) or string.lower() == 'null':
        return None
    try:
        # Unescape the string
        clean_string = string.replace('\""', '"')
        # Convert to dictionary
        data = json.loads(clean_string)
        data = json.loads(data)
        return data
    except json.JSONDecodeError:
        return None


def get_process_pattern(df: pd.DataFrame)->pd.DataFrame:
    """Get the process patterns from the log file."""
    # First, sort the DataFrame by timestamp to ensure actions are in the correct order
    df_sorted = df.sort_values(['process_id', 'timestamp'])

    # Group by process_id and create a sequence of actions in temporal order
    process_sequences = df_sorted.groupby('process_id')['action'] \
        .apply(lambda x: ' -> '.join(x)) \
        .reset_index(name='action_sequence')

    # Create a DataFrame with process id, action sequence, and frequency
    process_pattern_counts = process_sequences['action_sequence'].value_counts().reset_index()
    process_pattern_counts.columns = ['action_sequence', 'frequency']
    
    return process_pattern_counts

def get_unique_data_values(df: pd.DataFrame, keys: Dict) -> pd.DataFrame:
    # Get unique data values for specified keys
    # output a DataFrame with keys, their unique values, and ratio of unique values to total values
    unique_data = {}
    for key in keys:
        unique_values = df['data_values'].apply(parse_data).apply(lambda x: x.get(key) if isinstance(x, dict) else None)
        unique_data[key] = {
            'unique_values': unique_values.dropna().unique(),
            'ratio': unique_values.notna().sum() / len(df) if len(df) > 0 else 0
        }
    return pd.DataFrame(unique_data)

def get_log_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Get statistics from the log file."""
    stats = {
        'total events': len(df),
        'total process instances': df['process_id'].nunique(),
        'longest_process_length': df['process_id'].value_counts().max(),
        'shortest_process_length': df['process_id'].value_counts().min(),
        'average_process_length': df['process_id'].value_counts().mean()
    }
    
    return pd.DataFrame([stats])

def get_action_durations(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # create columns task_difficulty and satisfaction_score basing on the parse function
    
    df['task_difficulty'] = df['feedback'].apply(parse_data).apply(
        lambda x: x.get('task_difficulty') if isinstance(x, dict) else None
    )
    df['satisfaction_score'] = df['feedback'].apply(parse_data).apply(
        lambda x: x.get('satisfaction_score') if isinstance(x, dict) else None
    )
    
    # Sort by process and timestamp
    df = df.sort_values(['process_id', 'timestamp'])
    
    # Calculate duration until next action within the same process
    df['action_duration'] = df.groupby('process_id')['timestamp'].diff().abs().dt.total_seconds()
    
    # diff(-1) computes timestamp of next row minus current row within each process.
    # abs() because diff(-1) returns negative differences.
    
    # Convert duration from seconds to hours
    df['action_duration'] = df['action_duration'] / 3600

    # Group by action and aggregate statistics
    action_durations = df.groupby('action').agg(
        longest_duration=('action_duration', 'max'),
        shortest_duration=('action_duration', 'min'),
        average_duration=('action_duration', 'mean'),
        lowest_task_difficulty=('task_difficulty', 'min'),
        highest_task_difficulty=('task_difficulty', 'max'),
        average_task_difficulty=('task_difficulty', 'mean'),
        lowest_satisfaction_score=('satisfaction_score', 'min'),
        highest_satisfaction_score=('satisfaction_score', 'max'),
        average_satisfaction_score=('satisfaction_score', 'mean')
    ).reset_index()
    
    return action_durations

def get_agent_statistics(df: pd.DataFrame) -> pd.DataFrame:
    # Group by agent, role, and process_id and aggregate all actions as a list or joined string
    agent_stats = df.groupby(['agent', 'role', 'process_id'])['action'] \
        .apply(list) \
        .reset_index()
    
    # Rename the aggregated column
    agent_stats.columns = ['agent', 'role', 'process_id', 'actions_list']
    
    return agent_stats


def main():
    log_file = './logs/insurance_claim_process_06_new_new_new_new_3.csv'
    cols = ['process_id', 'action', 'timestamp', 'role', 'agent', 'external_values', 'data_values', 'feedback']
    df = pd.read_csv(log_file, header=None, names=cols)

    # Get log statistics
    log_stats = get_log_statistics(df)
    #print("Log Statistics:")
    #print(log_stats)

    # Get process patterns
    process_patterns = get_process_pattern(df)
    #print("\nProcess Patterns and their Frequencies:")
    #with pd.option_context('display.max_colwidth', None):
    #    print(process_patterns.head(10))  # Display the top 10 patterns for brevity

    # Get unique data values
    #print("\nUnique Data Values:")
    keys_of_interest = ['incident_description', 'damage', 'registration_number', 'vehicle']
    unique_data_values = get_unique_data_values(df, keys_of_interest)
    #print(unique_data_values)

    # write each df to a separate file
    log_stats.to_csv('./csv_files/log_statistics.csv', index=False)
    process_patterns.to_csv('./csv_files/process_patterns.csv', index=False)
    unique_data_values.to_csv('./csv_files/unique_data_values.csv', index=False)

    agent_stats = get_agent_statistics(df=df)
    agent_stats.to_csv('./csv_files/agent_statistics.csv', index=False)

    action_durations = get_action_durations(df=df)
    action_durations.to_csv('./csv_files/action_statistics.csv', index=False)
if __name__ == "__main__":
    main()