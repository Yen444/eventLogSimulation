### Helper function
from datetime import datetime
import random
import string

# generating random unique order ID
def generate_order_id(order_counter):
    # Generate 3 random letters
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))   
    # Generate 3 random digits
    digits = ''.join(random.choices(string.digits, k=2))
    # Combine the letters and digits
    random_id = f"{letters}{digits}_{order_counter}"
    return random_id

# Characters and names of potential customers

characters = [
    "You are a health-conscious individual, who carefully tracks nutritional intake and rarely indulges in fast food.",
    "You're a busy software engineer who values quick meals during lunch breaks.",
    "As a fitness enthusiast, you occasionally treat yourself to a cheat meal pizza after an intense workout in the evening.",
    "You're a food blogger always looking to try and review new pizza varieties.",
    "You're vegetarian who loves all kinds of food as long as it is vegetatian",
    "You work night shifts and often crave comfort food during late hours.",
    "You're a student, who prefer budget lunch or dinner",
    "You're a remote worker who enjoys breaking up the monotony of work with occasional food deliveries.",
    "You're an athlete with a high-metabolism diet who needs substantial, protein-rich meals.",
    "You're a budget-conscious graduate student who looks for affordable dining options.",
    "You have a large family and often order pizza for group meals and gatherings.",
    "You're a travel photographer who relies on quick meals between assignments.",
    "You're a teacher and doesn't have much time to cook. However, somebody need to prepare dinner for your children.",
    "You're a music teacher who uses meal times as a break between teaching sessions.",
    "You're a freelance writer who enjoys spontaneous lunch breaks.",
    "You're a retired chef who is very particular about food preparation and ingredients.",
    "You're an urban commuter who values convenient meal options during busy days.",
    "You're a startup entrepreneur who needs quick, energy-boosting meals.",
    "You're a part-time yoga instructor with specific dietary preferences.",
    "You're a marine biologist who occasionally treats themselves after long research days.",
    "You're a computer science student and you can have pizza anytime anywhere",
    "You're a mother of 3 kids. You don't want your children to have pizza but on rainy days, why not?",
    "You're housewife, if not for special accasion like family gathering or you're busy, you will always cook.",
    "You can eat anything but you don't want to spend 2 hours waiting for your lunch."
]
names = [
    "Alice", "Benjamin", "Charlotte", "Daniel", "Emily", "Felix", 
    "Grace", "Henry", "Isabella", "Jack", "Katherine", "Leo", 
    "Mia", "Nathan", "Olivia", "Patrick", "Quinn", "Ryan", 
    "Sophia", "Thomas", "Ursula", "Vincent", "William", "Zoe"
]

# generate randon unique customer profile
def generate_unique_customer(used_combinations):
    if len(used_combinations) >= len(names) * len(characters) * 20:  # Considering a range of distances
        raise ValueError("All unique combinations have been used. Cannot generate more unique customers.")
    
    # Choose a name, character, and distance
    name = random.choice(names)
    character = random.choice(characters)
    distance = str(random.randint(1, 20))  # Distance in km
    
    # Create a unique combination of name, distance, and character
    combination = (name, distance, character)
    
    # Add the combination to the used set to ensure uniqueness
    while combination in used_combinations:
        # If combination already exists, generate a new one
        name = random.choice(names)
        character = random.choice(characters)
        distance = str(random.randint(1, 20))  # Distance in km
        combination = (name, distance, character)
    
    used_combinations.append(combination)
    
    return {'name': combination[0], 'distance': combination[1], 'character': combination[2]}

# generate randon customer profile
def generate_random_customer():
    # Choose a name, character, and distance
    name = random.choice(names)
    character = random.choice(characters)
    distance = str(random.randint(1, 20))  # Distance in km
    
    # Create a combination of name, distance, and character
    combination = (name, distance, character)
    
    return {'name': combination[0], 'distance': combination[1], 'character': combination[2]}

# Generating random possible numper of order basing on time
import numpy as np
import datetime

# generate random number of orders
def generate_num_orders(timestamp):
    """
    Generate the number of orders arriving at a pizzeria within the next 10 minutes 
    based on the time of day using a Poisson distribution.

    Parameters:
        timestamp (datetime.datetime): The current timestamp.

    Returns:
        int: Number of orders arriving within the next 10 minutes.
    """
    hour = timestamp.hour

    # Define the average number of orders per minute (lambda) based on time of day
    arrival_rates = {
        11: 6/60, 12: 12/60, 13: 10/60, 14: 7/60,  # Lunch peak
        18: 10/60, 19: 11/60, 20: 6/60, 21: 5/60,  # Dinner peak
        8: 1/60, 9: 1/60, 10: 1.5/60,       # Morning low
        15: 1/60, 16: 3/60, 17: 5/60,       # Afternoon moderate
        22: 3/60, 23: 1/60,            # Late night low
    }
    
    # Default to a very low arrival rate for early morning (0:00 - 7:59) or other unspecified hours
    default_lambda = 0.2 if hour < 8 or hour > 23 else 1

    # Get arrival rate (λ) for the given hour
    lambda_val = arrival_rates.get(hour, default_lambda)

    # Number of orders per 10 minutes = 10 * arrival rate (λ)
    lambda_10min = lambda_val * 10  # orders per 10 minutes

    # Generate number of orders from Poisson distribution
    num_orders = np.random.poisson(lambda_10min)

    return num_orders


# Get latest states of uncompleted order

from typing import TypedDict, List, Dict

class Orderstate(TypedDict):
    customer_name: str
    distance: str
    pizza_type: str
    states: List[str]
    timestamps: List[str]

class CustomerInfo(TypedDict):
    name: str
    character: str
    distance: str
    
# state to action dict:
state2act_dict = {
    'oder confirmed': 'you confirmed',
    'oder being prepared': 'you are baking it and you started baking at',
    'oder being delivered': 'you re on the way to deliver it and you started delivering it at'
}
def state2act(state):
    return state2act_dict.get(state, state)
# get latest state of an order 
def get_latest_states(un_orders_dict: Dict[str, Orderstate]) -> str:
    result_lines = []
    
    for order_id, order in un_orders_dict.items():
        if order["states"] and order["timestamps"]:
            latest_state = order["states"][-1]
            latest_timestamp = order["timestamps"][-1]
            result_lines.append(
                f"Order ID {order_id} of {order['pizza_type']}, from {order['customer_name']}, {order['distance']} away, is {latest_state}, which means {state2act(latest_state)} at {latest_timestamp}"
            )
    
    return "\n".join(result_lines)