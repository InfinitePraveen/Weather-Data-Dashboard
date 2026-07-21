import pandas as pd
import numpy as np
from datetime import datetime
import os


class WeatherDataLoader:
    def __init__(self, file_path):
        """Initialize with path to weather data file"""
        self.file_path = file_path
        self.df = None
        self.locations = []
        self.load_data()

    def load_data(self):
        """Load and preprocess weather data"""
        # Try to detect file format
        if self.file_path.endswith(".csv"):
            self.df = pd.read_csv(self.file_path)
        elif self.file_path.endswith(".xlsx"):
            self.df = pd.read_excel(self.file_path)
        elif self.file_path.endswith(".json"):
            self.df = pd.read_json(self.file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV, Excel, or JSON.")

        self.preprocess_data()
        return self.df

    def preprocess_data(self):
        """Clean and prepare data for analysis"""
        # Identify columns based on your dataset structure
        # Location column
        if "Location" in self.df.columns:
            self.location_col = "Location"
            self.locations = self.df["Location"].unique().tolist()
            print(f"Found {len(self.locations)} unique locations: {self.locations}")

        # Date_Time column
        if "Date_Time" in self.df.columns:
            self.date_col = "Date_Time"
            self.df["Date_Time"] = pd.to_datetime(self.df["Date_Time"])
            self.df.set_index("Date_Time", inplace=True)
            self.df.sort_index(inplace=True)
            print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")

        # Weather variables - exactly matching your dataset
        self.weather_cols = []
        weather_mapping = {
            "Temperature_C": "Temperature (°C)",
            "Humidity_pct": "Humidity (%)",
            "Precipitation_mm": "Precipitation (mm)",
            "Wind_Speed_kmh": "Wind Speed (km/h)",
        }

        for col in weather_mapping.keys():
            if col in self.df.columns:
                self.weather_cols.append(col)

        # If Location exists, keep it for grouping
        if hasattr(self, "location_col"):
            # Reset index temporarily to handle location properly
            self.df.reset_index(inplace=True)
            self.df.set_index(["Location", self.df.index], inplace=True)
            self.df.index.names = ["Location", "Date_Time"]

        print(
            f"Loaded {len(self.df)} rows with {len(self.weather_cols)} weather variables"
        )
        print(f"Weather variables: {', '.join(self.weather_cols)}")

        # Check for missing values
        missing = self.df[self.weather_cols].isnull().sum()
        if missing.sum() > 0:
            print(f"Missing values detected:\n{missing[missing > 0]}")
            # Interpolate missing values
            self.df[self.weather_cols] = self.df.groupby("Location")[
                self.weather_cols
            ].transform(lambda x: x.interpolate(method="time"))

    def get_summary_stats(self, location=None):
        """Get summary statistics of weather data for specific location or all"""
        if location:
            data = self.get_location_data(location)
        else:
            data = self.df

        if len(self.weather_cols) > 0:
            stats = data[self.weather_cols].describe()
            stats.loc["missing"] = data[self.weather_cols].isna().sum()
            return stats
        return pd.DataFrame()

    def get_location_data(self, location):
        """Get data for a specific location"""
        if hasattr(self, "location_col"):
            return self.df.xs(location, level="Location")
        else:
            print("No location column found in dataset")
            return self.df

    def get_weather_data(self, location=None, columns=None):
        """Return weather data for specified location and columns"""
        if location:
            data = self.get_location_data(location)
        else:
            data = self.df

        if columns is None:
            columns = self.weather_cols
        return data[columns]

    def get_location_summary(self):
        """Get summary statistics per location"""
        if hasattr(self, "location_col"):
            summary = {}
            for location in self.locations:
                data = self.get_location_data(location)
                summary[location] = {
                    "count": len(data),
                    "date_range": f"{data.index.min()} to {data.index.max()}",
                    "avg_temperature": data["Temperature_C"].mean(),
                    "avg_humidity": data["Humidity_pct"].mean(),
                    "total_precipitation": data["Precipitation_mm"].sum(),
                    "avg_wind_speed": data["Wind_Speed_kmh"].mean(),
                }
            return pd.DataFrame(summary).T
        return pd.DataFrame()
