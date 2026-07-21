import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


class WeatherVisualizer:
    def __init__(self, data_loader):
        self.df = data_loader.df
        self.weather_cols = data_loader.weather_cols
        self.locations = (
            data_loader.locations if hasattr(data_loader, "locations") else []
        )
        self.loader = data_loader

    def plot_time_series(self, location=None, columns=None, figsize=(15, 10)):
        """Plot time series for weather variables"""
        if columns is None:
            columns = self.weather_cols

        if location:
            data = self.loader.get_location_data(location)
            title_suffix = f" - {location}"
        else:
            data = self.df
            title_suffix = " - All Locations"

        # Limit to 4 variables for readability
        cols_to_plot = columns[:4]

        fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=figsize)
        if len(cols_to_plot) == 1:
            axes = [axes]

        for i, col in enumerate(cols_to_plot):
            # Get data and resample for better visualization if too many points
            if len(data) > 10000:
                plot_data = data[col].resample("D").mean()
            else:
                plot_data = data[col]

            axes[i].plot(plot_data.index, plot_data.values, linewidth=1.5)
            axes[i].set_title(f"{col} Over Time{title_suffix}", fontsize=12)
            axes[i].set_xlabel("Date")
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_multiple_locations(self, column, figsize=(15, 8)):
        """Plot same variable for multiple locations"""
        if not self.locations or len(self.locations) == 0:
            print("No locations found in dataset")
            return

        if column not in self.weather_cols:
            print(f"Column '{column}' not found")
            return

        plt.figure(figsize=figsize)

        for location in self.locations:
            data = self.loader.get_location_data(location)
            if len(data) > 10000:
                plot_data = data[column].resample("D").mean()
            else:
                plot_data = data[column]
            plt.plot(plot_data.index, plot_data.values, label=location, linewidth=1.5)

        plt.title(f"{column} Comparison Across Locations")
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_seasonal_decomposition(self, column, location=None, period=365):
        """Plot seasonal decomposition of a time series"""
        if column not in self.weather_cols:
            print(f"Column '{column}' not found")
            return

        if location:
            data = self.loader.get_location_data(location)
        else:
            data = self.df

        # Ensure data is complete
        series = data[column].dropna()

        # Resample to daily if too many points
        if len(series) > 5000:
            series = series.resample("D").mean()

        # Perform decomposition
        try:
            decomposition = seasonal_decompose(series, model="additive", period=period)

            fig, axes = plt.subplots(4, 1, figsize=(12, 10))

            decomposition.observed.plot(ax=axes[0])
            axes[0].set_title("Observed")
            axes[0].grid(True, alpha=0.3)

            decomposition.trend.plot(ax=axes[1])
            axes[1].set_title("Trend")
            axes[1].grid(True, alpha=0.3)

            decomposition.seasonal.plot(ax=axes[2])
            axes[2].set_title("Seasonal")
            axes[2].grid(True, alpha=0.3)

            decomposition.resid.plot(ax=axes[3])
            axes[3].set_title("Residual")
            axes[3].grid(True, alpha=0.3)

            plt.suptitle(
                f'Seasonal Decomposition of {column}{f" - {location}" if location else ""}'
            )
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Could not perform decomposition: {e}")
            print("Try reducing the period parameter or using a different column")

    def create_correlation_heatmap(self, location=None):
        """Create correlation heatmap of weather variables"""
        if len(self.weather_cols) < 2:
            print("Need at least 2 weather variables for correlation")
            return

        if location:
            data = self.loader.get_location_data(location)
            title = f"Correlation Heatmap - {location}"
        else:
            data = self.df
            title = "Correlation Heatmap - All Locations"

        corr_matrix = data[self.weather_cols].corr()

        # Rename columns for better display
        display_names = {
            "Temperature_C": "Temperature",
            "Humidity_pct": "Humidity",
            "Precipitation_mm": "Precipitation",
            "Wind_Speed_kmh": "Wind Speed",
        }

        corr_matrix = corr_matrix.rename(columns=display_names, index=display_names)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".2f",
            square=True,
            linewidths=1,
        )
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_daily_aggregates(self, column, location=None):
        """Plot daily aggregates (mean, min, max)"""
        if column not in self.weather_cols:
            print(f"Column '{column}' not found")
            return

        if location:
            data = self.loader.get_location_data(location)
            title_suffix = f" - {location}"
        else:
            data = self.df
            title_suffix = " - All Locations"

        daily_data = data[column].resample("D").agg(["mean", "min", "max"])

        plt.figure(figsize=(14, 6))
        plt.plot(daily_data.index, daily_data["mean"], label="Mean", linewidth=1.5)
        plt.fill_between(
            daily_data.index,
            daily_data["min"],
            daily_data["max"],
            alpha=0.2,
            label="Min/Max Range",
        )
        plt.title(f"Daily {column}{title_suffix}")
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def create_interactive_dashboard(self, location=None):
        """Create an interactive Plotly dashboard"""
        if location:
            data = self.loader.get_location_data(location)
            title = f"Weather Dashboard - {location}"
        else:
            data = self.df
            title = "Weather Dashboard - All Locations"

        if len(self.weather_cols) < 1:
            print("No weather data available")
            return

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Temperature (°C)",
                "Humidity (%)",
                "Precipitation (mm)",
                "Wind Speed (km/h)",
                "Distribution - Temperature",
                "Distribution - Humidity",
            ),
        )

        # Color mapping for variables
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        # Add time series plots for first 4 weather variables
        for i, col in enumerate(self.weather_cols[:4]):
            row = i // 2 + 1
            col_pos = i % 2 + 1

            # Resample if too many points
            if len(data) > 10000:
                plot_data = data[col].resample("D").mean()
            else:
                plot_data = data[col]

            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data.values,
                    mode="lines",
                    name=col,
                    line=dict(color=colors[i % len(colors)]),
                ),
                row=row,
                col=col_pos,
            )

        # Add distribution plots
        if len(self.weather_cols) >= 1:
            fig.add_trace(
                go.Histogram(
                    x=data[self.weather_cols[0]],
                    name="Temp Distribution",
                    nbinsx=30,
                    marker_color="lightblue",
                ),
                row=3,
                col=1,
            )

        if len(self.weather_cols) >= 2:
            fig.add_trace(
                go.Histogram(
                    x=data[self.weather_cols[1]],
                    name="Humidity Distribution",
                    nbinsx=30,
                    marker_color="lightgreen",
                ),
                row=3,
                col=2,
            )

        # Update layout
        fig.update_layout(
            height=900, showlegend=True, title_text=title, template="plotly_white"
        )
        fig.show()

    def plot_location_comparison(self, column):
        """Create boxplot comparing a variable across locations"""
        if column not in self.weather_cols:
            print(f"Column '{column}' not found")
            return

        if not self.locations or len(self.locations) == 0:
            print("No locations found")
            return

        plt.figure(figsize=(12, 6))

        # Prepare data for boxplot
        data_to_plot = []
        location_names = []

        for location in self.locations:
            data = self.loader.get_location_data(location)
            data_to_plot.append(data[column].values)
            location_names.append(location)

        bp = plt.boxplot(data_to_plot, labels=location_names, patch_artist=True)

        # Color the boxes
        for box in bp["boxes"]:
            box.set_facecolor("lightblue")
            box.set_alpha(0.7)

        plt.title(f"{column} Distribution by Location")
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_monthly_trends(self, column, location=None):
        """Plot monthly averages and trends"""
        if column not in self.weather_cols:
            print(f"Column '{column}' not found")
            return

        if location:
            data = self.loader.get_location_data(location)
            title_suffix = f" - {location}"
        else:
            data = self.df
            title_suffix = " - All Locations"

        monthly_avg = data[column].resample("M").mean()

        plt.figure(figsize=(14, 6))
        plt.plot(monthly_avg.index, monthly_avg.values, marker="o", linewidth=2)
        plt.title(f"Monthly Average {column}{title_suffix}")
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Return monthly statistics
        return monthly_avg.describe()
