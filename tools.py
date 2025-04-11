from langchain_core.tools import tool
from datetime import datetime


@tool
def get_weather(timestamp):
    
    """Get weather condition given timestamp"""

    dt = datetime.fromisoformat(timestamp)
    hour = dt.hour
    if 9 <= hour < 12:
        return "cold"
    elif 12 <= hour < 16:
        return "rainy and stormy"
    elif 16 <= hour < 19:
        return "rainy and windy"
    elif 19 <= hour < 22:
        return "heavily rainy"
    else:
        return "no weather data for this hour"

@tool
def get_weather_1(timestamp):
    
    """Get weather condition given timestamp"""

    dt = datetime.fromisoformat(timestamp)
    hour = dt.hour
    if 9 <= hour < 12:
        return "chilly"
    elif 12 <= hour < 16:
        return "mild"
    elif 16 <= hour < 19:
        return "cool"
    elif 19 <= hour < 22:
        return "cold"
    else:
        return "no weather data for this hour"

@tool
def get_traffic(timestamp, distance):

    """Get traffic condition given timestamp and distance"""

    dt = datetime.fromisoformat(timestamp)
    hour = dt.hour

    is_peak_hour = (12 <= hour <= 14) or (17 <= hour <= 19)

    if distance <= 10:
        return "Busy traffic" if is_peak_hour else "Normal traffic"
    else:
        if is_peak_hour:
            return "Busy traffic in the city center. It may improve outside the city, but the distance is quite long."
        else:
            return "Normal traffic"

from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    """Get weather condition basing on given timestamp"""
    timestamp: str = Field(description='Given timestamp')
class GetTraffic(BaseModel):
    """Get traffic condition given timestamp and distance"""
    timestamp: str = Field(description='Given timestamp')
    distance: float = Field(description='Given distance')