import random
import string 

# Function to read .bpmn file and return a single string
def read_bpmn_file(bpmn_path: str) -> str:
    """
    Reads a BPMN file and returns its content as a single string.

    Args:
        bpmn_path (str): Path to the BPMN file.

    Returns:
        str: The BPMN file content as a string.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        UnicodeDecodeError: If the file cannot be decoded with UTF-8 encoding.
        IOError: For general I/O errors while reading the file.
    """
    try:
        with open(bpmn_path, 'r', encoding='utf-8') as file:
            model_string = file.read()
        return model_string
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at '{bpmn_path}' was not found.")
    except UnicodeDecodeError:
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "Failed to decode BPMN file with UTF-8 encoding.")
    except IOError as e:
        raise IOError(f"An error occurred while reading the BPMN file: {e}")

# Function generate a random process ID of 3 letters and 2 digits and this ID should be unique
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



import numpy as np

def simulate_claims_with_interarrival_time(mean_interarrival_time, duration_minutes=60, seed=None):
    """
    Simulate claim arrival times using inter-arrival times.
    
    Parameters:
    - mean_interarrival_time: average time between claims (in minutes)
    - duration_minutes: total simulation time (in minutes)
    - seed: optional random seed
    
    Returns:
    - arrival_times: list of claim arrival times (in minutes)
    """
    if seed is not None:
        np.random.seed(seed)
    
    arrival_times = []
    current_time = 0.0

    while current_time < duration_minutes:
        interarrival = np.random.exponential(scale=mean_interarrival_time)
        current_time += interarrival
        if current_time < duration_minutes:
            arrival_times.append(current_time)

    return arrival_times

def simulate_number_of_arrivals(rate_per_minute, duration_minutes, seed=None):
    """
    Simulate how many events (e.g., claims) arrive in a fixed time window.

    Parameters:
    - rate_per_minute: expected number of arrivals per minute (λ)
    - duration_minutes: length of the time window to simulate
    - seed: optional seed for reproducibility

    Returns:
    - num_arrivals: integer, number of arrivals in the window
    """
    if seed is not None:
        np.random.seed(seed)
    
    expected_arrivals = rate_per_minute * duration_minutes
    num_arrivals = np.random.poisson(expected_arrivals)
    
    return num_arrivals
    



