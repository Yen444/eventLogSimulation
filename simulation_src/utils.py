import random
import string

import os
import json
from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Optional, Any

def generate_process_id(used_ids:list) -> str:
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

def generate_agent_id(used_ids:list) -> str:
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
    # Parse ISO 8601 timestamp
    dt = datetime.fromisoformat(timestamp_str)

    hour = dt.strftime('%I:%M %p').lstrip('0').lower()      # e.g., '9:30 am'
    day_name = dt.strftime('%A')                            # e.g., 'Monday'
    day_suffix = get_day_with_suffix(dt.day)                # e.g., '1st'
    month_name = dt.strftime('%B')                          # e.g., 'August'
    month_with_day = f"{month_name} {day_suffix}"           # e.g., 'August 1st'

    return f"at {hour} on {day_name} in {month_with_day}"
