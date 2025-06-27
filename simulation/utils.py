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
import matplotlib.pyplot as plt

def simulate_requests_per_hour(lambda_rate, num_hours, plot=False):
    """
    Simulates the number of requests arriving each hour using a Poisson distribution.

    Parameters:
    - lambda_rate (float): Average number of requests per hour.
    - num_hours (int): Number of hours to simulate.
    - plot (bool): If True, shows a line plot of the results.

    Returns:
    - requests (np.ndarray): Array with the number of requests per hour.
    """
    # Sample from the Poisson distribution
    requests = np.random.poisson(lam=lambda_rate, size=num_hours)

    # Print hourly results
    for hour, count in enumerate(requests, start=1):
        print(f"Hour {hour}: {count} requests")

    # Optional: plot the simulation
    if plot:
        plt.plot(range(1, num_hours + 1), requests, marker='o')
        plt.title("Simulated Requests Per Hour")
        plt.xlabel("Hour")
        plt.ylabel("Number of Requests")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return requests

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import calendar

# Average requests per month (index 0 = Jan)
monthly_rates = [1.0, 0.9, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 1.1, 1.0, 0.95, 0.9]

# Average requests per weekday (0=Mon, ..., 6=Sun)
weekday_rates = [1.0, 1.05, 1.1, 1.1, 1.2, 0.9, 0.8]

# Average requests per hour (0–23)
hourly_rates = [0.5, 0.3, 0.2, 0.2, 0.3, 0.6, 1.0, 1.5, 2.0, 2.2, 2.0, 1.8,
                1.7, 1.6, 1.8, 2.0, 2.3, 2.5, 2.0, 1.5, 1.0, 0.8, 0.6, 0.4]

# Base rate (can be scaled up later)
BASE_RATE_HOUR = 0.01 # there is one request every 100 hours on average


# For the P2P EC Process

hour_rates = {
    3: 9, 4: 25, 5: 40, 6: 69, 7: 86, 8: 97, 9: 83, 10: 83,
    11: 75, 12: 68, 13: 76, 14: 41, 15: 20, 16: 7, 17: 2,
    18: 1, 19: 1, 20: 13, 21: 26, 22: 7
}
day_rates = {0: 135, 1: 121, 2: 201, 3: 193, 4: 158, 5: 18, 6: 3}
month_rates = {
    1: 62, 2: 125, 3: 92, 4: 55, 5: 60, 6: 77,
    7: 54, 8: 58, 9: 77, 10: 74, 11: 66, 12: 29
}

def get_lambda(timestamp, base_rate, monthly_rate, weekday_rate, hourly_rate):
    month_idx = timestamp.month - 1
    weekday_idx = timestamp.weekday()
    hour_idx = timestamp.hour
    return base_rate * monthly_rate[month_idx] * weekday_rate[weekday_idx] * hourly_rate[hour_idx]

def simulate_arrivals(start_time, base_rate, monthly_rate, weekday_rate, hourly_rate, max_events=10):
    t = start_time
    arrivals = []
    λ_max = base_rate * max(monthly_rate) * max(weekday_rate) * max(hourly_rate)

    while len(arrivals) < max_events:
        # Step 1: Sample time to next candidate event
        u = np.random.uniform()
        delta = -np.log(u) / λ_max
        t_candidate = t + timedelta(hours=delta)

        # Step 2: Calculate λ(t_candidate)
        λ_t = get_lambda(t_candidate, base_rate, monthly_rate, weekday_rate, hourly_rate)

        # Step 3: Accept or reject
        u2 = np.random.uniform()
        if u2 <= λ_t / λ_max:
            arrivals.append((t_candidate, λ_t))
            t = t_candidate  # Move time forward only on accept
        else:
            t = t_candidate  # Still move forward to candidate time

    return arrivals



"""start_time = datetime(2025, 1, 1)
arrivals = simulate_arrivals(
    start_time,
    base_rate=BASE_RATE_HOUR,
    monthly_rate=monthly_rates,
    weekday_rate=weekday_rates,
    hourly_rate=hourly_rates
)
# calculate mean inter-arrival times
if len(arrivals) > 1:
    inter_arrival_times = [(arrivals[i][0] - arrivals[i-1][0]).total_seconds() / 3600 for i in range(1, len(arrivals))]
    mean_inter_arrival = np.mean(inter_arrival_times)
    print(f"Mean inter-arrival time: {mean_inter_arrival:.2f} hours")
    print(inter_arrival_times)"""