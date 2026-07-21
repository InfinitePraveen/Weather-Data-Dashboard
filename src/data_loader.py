import pandas as pd
import numpy as np
from datetime import datetime
import os


class WeatherDataLoader:
    def __init__(self, file_path):
        """Initialize with path to weather data file"""
        self.file_path = file_path
        self.df = None
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
        # Detect date column (common names)
        date_columns = [
            "date",
            "datetime",
            "timestamp",
            "time",
            "day",
            "Date",
            "DateTime",
        ]
        date_col = None

        for col in date_columns:
            if col in self.df.columns:
                date_col = col
                break

        # If no standard date column, try to infer
        if date_col is None:
            for col in self.df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    date_col = col
                    break

        if date_col:
            # Convert to datetime
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors="coerce")
            self.df.set_index(date_col, inplace=True)
            self.df.sort_index(inplace=True)

        # Identify numeric columns for weather data
        weather_columns = [
            "temperature",
            "temp",
            "humidity",
            "pressure",
            "wind_speed",
            "precipitation",
            "rainfall",
            "cloud_cover",
        ]

        self.weather_cols = []
        for col in weather_columns:
            for df_col in self.df.columns:
                if col in df_col.lower():
                    self.weather_cols.append(df_col)
                    break

        # If no weather columns found, use all numeric columns
        if not self.weather_cols:
            self.weather_cols = self.df.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        # Handle missing values
        if len(self.weather_cols) > 0:
            self.df[self.weather_cols] = self.df[self.weather_cols].interpolate(
                method="time"
            )

        print(
            f"Loaded data with {len(self.df)} rows and {len(self.weather_cols)} weather variables"
        )
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")

    def get_summary_stats(self):
        """Get summary statistics of weather data"""
        if len(self.weather_cols) > 0:
            stats = self.df[self.weather_cols].describe()
            stats.loc["missing"] = self.df[self.weather_cols].isna().sum()
            return stats
        return pd.DataFrame()

    def get_weather_data(self, columns=None):
        """Return weather data for specified columns or all weather columns"""
        if columns is None:
            columns = self.weather_cols
        return self.df[columns]
