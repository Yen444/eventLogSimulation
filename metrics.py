"""
Module for calculating various metrics between event logs.
This module contains all the metric calculation functions for evaluating
the similarity and distance between generated and real event logs.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from log_distance_measures.config import EventLogIDs, DEFAULT_CSV_IDS, AbsoluteTimestampType, discretize_to_hour
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance


def prepare_log_for_metrics(log_df, event_log_ids):
    """
    Prepare a log dataframe for metric calculations by setting up timestamps.
    
    Args:
        log_df: DataFrame with event log data
        event_log_ids: EventLogIDs object with column mappings
        
    Returns:
        DataFrame with properly formatted timestamps
    """
    log_copy = log_df.copy()
    
    # Ensure end_time is properly formatted
    log_copy[event_log_ids.end_time] = pd.to_datetime(
        log_copy[event_log_ids.end_time],
        utc=True,
        format='mixed'
    )
    
    # Set start_time if it doesn't exist or needs formatting
    if event_log_ids.start_time not in log_copy.columns or log_copy[event_log_ids.start_time].isna().all():
        log_copy[event_log_ids.start_time] = log_copy[event_log_ids.end_time] - timedelta(minutes=30)
    else:
        log_copy[event_log_ids.start_time] = pd.to_datetime(
            log_copy[event_log_ids.start_time],
            utc=True,
            format='mixed'
        )
    
    return log_copy


def calculate_control_flow_distance(log1, log2, log1_ids=None, log2_ids=None):
    """
    Calculate control flow distance between two event logs.
    
    Args:
        log1, log2: Event log DataFrames
        log1_ids, log2_ids: EventLogIDs objects (default: DEFAULT_CSV_IDS)
        
    Returns:
        float: Control flow distance
    """
    if log1_ids is None:
        log1_ids = DEFAULT_CSV_IDS
    if log2_ids is None:
        log2_ids = DEFAULT_CSV_IDS
    
    return control_flow_log_distance(log1, log1_ids, log2, log2_ids)


def calculate_ngram_distance(log1, log2, n=3, log1_ids=None, log2_ids=None):
    """
    Calculate n-gram distribution distance between two event logs.
    
    Args:
        log1, log2: Event log DataFrames
        n: N-gram size (default: 3)
        log1_ids, log2_ids: EventLogIDs objects (default: DEFAULT_CSV_IDS)
        
    Returns:
        float: N-gram distribution distance
    """
    if log1_ids is None:
        log1_ids = DEFAULT_CSV_IDS
    if log2_ids is None:
        log2_ids = DEFAULT_CSV_IDS
    
    return n_gram_distribution_distance(log1, log1_ids, log2, log2_ids, n=n)


def calculate_absolute_event_distribution_distance(log1, log2, log1_ids=None, log2_ids=None, 
                                                 discretize_type=AbsoluteTimestampType.END,
                                                 discretize_func=discretize_to_hour):
    """
    Calculate absolute event distribution distance between two event logs.
    
    Args:
        log1, log2: Event log DataFrames
        log1_ids, log2_ids: EventLogIDs objects (default: DEFAULT_CSV_IDS)
        discretize_type: Which timestamps to consider
        discretize_func: Function to discretize timestamps
        
    Returns:
        float: Absolute event distribution distance
    """
    if log1_ids is None:
        log1_ids = DEFAULT_CSV_IDS
    if log2_ids is None:
        log2_ids = DEFAULT_CSV_IDS
    
    return absolute_event_distribution_distance(
        log1, log1_ids, log2, log2_ids,
        discretize_type=discretize_type,
        discretize_event=discretize_func
    )


def calculate_case_arrival_distribution_distance(log1, log2, log1_ids=None, log2_ids=None,
                                               discretize_func=discretize_to_hour):
    """
    Calculate case arrival distribution distance between two event logs.
    
    Args:
        log1, log2: Event log DataFrames
        log1_ids, log2_ids: EventLogIDs objects (default: DEFAULT_CSV_IDS)
        discretize_func: Function to discretize timestamps
        
    Returns:
        float: Case arrival distribution distance
    """
    if log1_ids is None:
        log1_ids = DEFAULT_CSV_IDS
    if log2_ids is None:
        log2_ids = DEFAULT_CSV_IDS
    
    return case_arrival_distribution_distance(
        log1, log1_ids, log2, log2_ids,
        discretize_event=discretize_func
    )


def calculate_circadian_event_distribution_distance(log1, log2, log1_ids=None, log2_ids=None,
                                                  discretize_type=AbsoluteTimestampType.END):
    """
    Calculate circadian event distribution distance between two event logs.
    
    Args:
        log1, log2: Event log DataFrames
        log1_ids, log2_ids: EventLogIDs objects (default: DEFAULT_CSV_IDS)
        discretize_type: Which timestamps to consider
        
    Returns:
        float: Circadian event distribution distance
    """
    if log1_ids is None:
        log1_ids = DEFAULT_CSV_IDS
    if log2_ids is None:
        log2_ids = DEFAULT_CSV_IDS
    
    return circadian_event_distribution_distance(
        log1, log1_ids, log2, log2_ids,
        discretize_type=discretize_type
    )


def calculate_relative_event_distribution_distance(log1, log2, log1_ids=None, log2_ids=None,
                                                 discretize_type=AbsoluteTimestampType.END,
                                                 discretize_func=discretize_to_hour):
    """
    Calculate relative event distribution distance between two event logs.
    
    Args:
        log1, log2: Event log DataFrames
        log1_ids, log2_ids: EventLogIDs objects (default: DEFAULT_CSV_IDS)
        discretize_type: Which timestamps to consider
        discretize_func: Function to discretize timestamps
        
    Returns:
        float: Relative event distribution distance
    """
    if log1_ids is None:
        log1_ids = DEFAULT_CSV_IDS
    if log2_ids is None:
        log2_ids = DEFAULT_CSV_IDS
    
    return relative_event_distribution_distance(
        log1, log1_ids, log2, log2_ids,
        discretize_type=discretize_type,
        discretize_event=discretize_func
    )


def calculate_cycle_time_distribution_distance(log1, log2, log1_ids=None, log2_ids=None,
                                             bin_size=pd.Timedelta(hours=1)):
    """
    Calculate cycle time distribution distance between two event logs.
    
    Args:
        log1, log2: Event log DataFrames
        log1_ids, log2_ids: EventLogIDs objects (default: DEFAULT_CSV_IDS)
        bin_size: Time bin size for discretization
        
    Returns:
        float: Cycle time distribution distance
    """
    if log1_ids is None:
        log1_ids = DEFAULT_CSV_IDS
    if log2_ids is None:
        log2_ids = DEFAULT_CSV_IDS
    
    return cycle_time_distribution_distance(
        log1, log1_ids, log2, log2_ids,
        bin_size=bin_size
    )


def calculate_circadian_workforce_distribution_distance(log1, log2, log1_ids=None, log2_ids=None):
    """
    Calculate circadian workforce distribution distance between two event logs.
    
    Args:
        log1, log2: Event log DataFrames
        log1_ids, log2_ids: EventLogIDs objects (default: DEFAULT_CSV_IDS)
        
    Returns:
        float: Circadian workforce distribution distance
    """
    if log1_ids is None:
        log1_ids = DEFAULT_CSV_IDS
    if log2_ids is None:
        log2_ids = DEFAULT_CSV_IDS
    
    return circadian_workforce_distribution_distance(log1, log1_ids, log2, log2_ids)


def calculate_all_metrics(log1, log2, log1_ids=None, log2_ids=None, verbose=True):
    """
    Calculate all available metrics between two event logs.
    
    Args:
        log1, log2: Event log DataFrames
        log1_ids, log2_ids: EventLogIDs objects (default: DEFAULT_CSV_IDS)
        verbose: Whether to print results
        
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    if log1_ids is None:
        log1_ids = DEFAULT_CSV_IDS
    if log2_ids is None:
        log2_ids = DEFAULT_CSV_IDS
    
    # Prepare logs for metric calculations
    prepared_log1 = prepare_log_for_metrics(log1, log1_ids)
    prepared_log2 = prepare_log_for_metrics(log2, log2_ids)
    
    metrics = {}
    
    # Control flow metrics
    metrics['control_flow_distance'] = calculate_control_flow_distance(
        prepared_log1, prepared_log2, log1_ids, log2_ids
    )
    
    # N-gram distribution metrics
    metrics['2gram_distance'] = calculate_ngram_distance(
        prepared_log1, prepared_log2, n=2, log1_ids=log1_ids, log2_ids=log2_ids
    )
    metrics['3gram_distance'] = calculate_ngram_distance(
        prepared_log1, prepared_log2, n=3, log1_ids=log1_ids, log2_ids=log2_ids
    )
    
    # Temporal distribution metrics
    metrics['absolute_event_distribution_distance'] = calculate_absolute_event_distribution_distance(
        prepared_log1, prepared_log2, log1_ids, log2_ids
    )
    
    metrics['case_arrival_distribution_distance'] = calculate_case_arrival_distribution_distance(
        prepared_log1, prepared_log2, log1_ids, log2_ids
    )
    
    metrics['circadian_event_distribution_distance'] = calculate_circadian_event_distribution_distance(
        prepared_log1, prepared_log2, log1_ids, log2_ids
    )
    
    metrics['relative_event_distribution_distance'] = calculate_relative_event_distribution_distance(
        prepared_log1, prepared_log2, log1_ids, log2_ids
    )
    
    metrics['cycle_time_distribution_distance'] = calculate_cycle_time_distribution_distance(
        prepared_log1, prepared_log2, log1_ids, log2_ids
    )
    
    metrics['circadian_workforce_distribution_distance'] = calculate_circadian_workforce_distribution_distance(
        prepared_log1, prepared_log2, log1_ids, log2_ids
    )
    
    if verbose:
        print("\n=== METRIC RESULTS ===")
        for metric_name, value in metrics.items():
            print(f'{metric_name}: {value}')
    
    return metrics


def compare_logs_with_baseline(real_log, generated_log, baseline_log, log_ids=None, verbose=True):
    """
    Compare generated log with real log and baseline (e.g., PLG2) log.
    
    Args:
        real_log: Real event log DataFrame
        generated_log: Generated event log DataFrame  
        baseline_log: Baseline event log DataFrame (e.g., PLG2)
        log_ids: EventLogIDs object (default: DEFAULT_CSV_IDS)
        verbose: Whether to print results
        
    Returns:
        dict: Dictionary containing comparison metrics
    """
    if log_ids is None:
        log_ids = DEFAULT_CSV_IDS
    
    results = {}
    
    if verbose:
        print("Calculating metrics: Real vs Generated log...")
    results['real_vs_generated'] = calculate_all_metrics(
        real_log, generated_log, log_ids, log_ids, verbose=False
    )
    
    if verbose:
        print("Calculating metrics: Real vs Baseline log...")
    results['real_vs_baseline'] = calculate_all_metrics(
        real_log, baseline_log, log_ids, log_ids, verbose=False
    )
    
    if verbose:
        print("\n=== COMPARISON RESULTS ===")
        print("Real vs Generated:")
        for metric, value in results['real_vs_generated'].items():
            print(f"  {metric}: {value}")
        
        print("\nReal vs Baseline:")
        for metric, value in results['real_vs_baseline'].items():
            print(f"  {metric}: {value}")
        
        print("\n=== IMPROVEMENT ANALYSIS ===")
        for metric in results['real_vs_generated'].keys():
            gen_value = results['real_vs_generated'][metric]
            baseline_value = results['real_vs_baseline'][metric]
            improvement = baseline_value - gen_value
            improvement_pct = (improvement / baseline_value) * 100 if baseline_value != 0 else 0
            
            status = "BETTER" if improvement > 0 else "WORSE" if improvement < 0 else "SAME"
            print(f"  {metric}: {status} ({improvement:+.4f}, {improvement_pct:+.1f}%)")
    
    return results