import random
import string
import numpy as np
from datetime import timedelta

import os
import json
from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Optional, Any

def generate_random_process_id(used_ids:list) -> str:
    """
    Generates a unique process ID consisting of 3 letters and 2 digits.

    Args:
        used_ids (list): List of already used process IDs.

    Returns:
        str: A unique process ID.
    """
    while True:
        letters = ''.join(random.choices(string.ascii_uppercase, k=3))
        digits = ''.join(random.choices(string.digits, k=2))
        process_id = f"{letters}{digits}"
        if process_id not in used_ids:
            return process_id

def generate_random_agent_id(used_ids:list) -> str:
    """
    Generates a unique Agent ID consisting of 5 digits.

    Args:
        used_ids (list): List of already used process IDs.

    Returns:
        str: A unique Agent ID.
    """
    while True:
        agent_id = ''.join(random.choices(string.digits, k=5))
        if agent_id not in used_ids:
            return agent_id

def create_dynamic_model(model_name: str, fields: Dict[str, Any]) -> BaseModel:
    """
    Create a dynamic Pydantic model with the given name and fields.

    Args:
        model_name (str): Name of the model.
        fields (Dict[str, Any]): Dictionary of field names and their description.

    Returns:
        BaseModel: A Pydantic model class.
    """
    fields = {field: (str, Field(..., description=description)) for field, description in fields.items()}
    return create_model(model_name, **fields)

from datetime import datetime

def get_day_with_suffix(day):
    if 11 <= day <= 13:
        return f"{day}th"
    suffixes = {1: 'st', 2: 'nd', 3: 'rd'}
    return f"{day}{suffixes.get(day % 10, 'th')}"

def extract_time_parts(timestamp_str):
    # Handle empty or None timestamp strings
    if not timestamp_str or timestamp_str.strip() == '':
        return "at an unspecified time"
    
    try:
        # Parse ISO 8601 timestamp
        dt = datetime.fromisoformat(timestamp_str)

        hour = dt.strftime('%I:%M %p').lstrip('0').lower()      # e.g., '9:30 am'
        day_name = dt.strftime('%A')                            # e.g., 'Monday'
        day_suffix = get_day_with_suffix(dt.day)                # e.g., '1st'
        month_name = dt.strftime('%B')                          # e.g., 'August'
        month_with_day = f"{month_name} {day_suffix}"           # e.g., 'August 1st'

        return f"at {hour} on {day_name} in {month_with_day}"
    except ValueError as e:
        # Handle malformed timestamp strings
        return timestamp_str



def get_lambda(timestamp, base_rate, monthly_rate, weekday_rate, hourly_rate):
    month_idx = timestamp.month - 1
    weekday_idx = timestamp.weekday()
    hour_idx = timestamp.hour
    return base_rate * monthly_rate[month_idx] * weekday_rate[weekday_idx] * hourly_rate[hour_idx]

def simulate_arrivals(start_time, base_rate, monthly_rate, weekday_rate, hourly_rate,
                      max_events=None, end_time=None):
    """
    Simulates a non-homogeneous Poisson process using the thinning algorithm.

    Parameters:
    - start_time: datetime, simulation start
    - base_rate: float, base hourly rate
    - monthly_rate: list of 12 floats (Jan=0 to Dec=11)
    - weekday_rate: list of 7 floats (Mon=0 to Sun=6)
    - hourly_rate: list of 24 floats (hour=0 to 23)
    - max_events: int, stop after this many accepted events (optional)
    - end_time: datetime, stop when this time is reached (optional)

    Returns:
    - List of tuples (arrival_time, λ_at_time)
    """
    if max_events is None and end_time is None:
        raise ValueError("Either max_events or end_time must be provided.")

    # add extra buffer to max_events
    max_events = max_events + max_events/5

    t = datetime.fromisoformat(start_time)
    arrivals = []

    # Compute maximum lambda for thinning
    λ_max = base_rate * max(monthly_rate) * max(weekday_rate) * max(hourly_rate)

    while True:
        # Stop if end_time is reached
        if end_time is not None and t >= end_time:
            break

        # Stop if enough events are collected
        if len(arrivals) >= max_events:
            break

        # Step 1: sample time until next candidate event
        u = np.random.uniform()
        delta = -np.log(u) / λ_max
        t_candidate = t + timedelta(hours=delta)

        # Stop if candidate is beyond end_time
        if end_time is not None and t_candidate > end_time:
            break

        # Step 2: evaluate actual lambda at t_candidate
        λ_t = get_lambda(t_candidate, base_rate, monthly_rate, weekday_rate, hourly_rate)

        # Step 3: accept or reject the candidate
        u2 = np.random.uniform()
        if u2 <= λ_t / λ_max:
            arrivals.append((t_candidate, λ_t))
            t = t_candidate  # move time forward on accept
        else:
            t = t_candidate  # still move forward even if rejected
    
    if isinstance(arrivals[0], tuple):
        times = np.array([a[0].timestamp() for a in arrivals])
    else:
        times = np.array([a.timestamp() for a in arrivals])

    # Compute differences and convert from seconds to hours
    inter_arrival_time = np.mean(np.diff(times) / 3600)

    return arrivals, inter_arrival_time


def main():

    pass

if __name__ == "__main__":
    main()
