"""
Weather Data Dashboard Package
A comprehensive tool for weather data analysis and visualization
"""

from .data_loader import WeatherDataLoader
from .visualizations import WeatherVisualizer
from .dashboard import WeatherDashboard

__version__ = "1.0.0"
__all__ = ["WeatherDataLoader", "WeatherVisualizer", "WeatherDashboard"]
