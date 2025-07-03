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