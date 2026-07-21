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

    def plot_time_series(self, columns=None, figsize=(15, 8)):
        """Plot time series for weather variables"""
        if columns is None:
            columns = self.weather_cols[:3]  # Limit to first 3 for readability

        plt.figure(figsize=figsize)

        for i, col in enumerate(columns[:3], 1):  # Max 3 subplots
            plt.subplot(3, 1, i)
            plt.plot(self.df.index, self.df[col], linewidth=1.5)
            plt.title(f"{col} Over Time", fontsize=12)
            plt.xlabel("Date")
            plt.ylabel(col)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_seasonal_decomposition(self, column, period=365):
        """Plot seasonal decomposition of a time series"""
        if column not in self.df.columns:
            print(f"Column '{column}' not found")
            return

        # Ensure data is complete
        data = self.df[column].dropna()

        # Perform decomposition
        try:
            decomposition = seasonal_decompose(data, model="additive", period=period)

            fig, axes = plt.subplots(4, 1, figsize=(12, 10))

            decomposition.observed.plot(ax=axes[0])
            axes[0].set_title("Observed")

            decomposition.trend.plot(ax=axes[1])
            axes[1].set_title("Trend")

            decomposition.seasonal.plot(ax=axes[2])
            axes[2].set_title("Seasonal")

            decomposition.resid.plot(ax=axes[3])
            axes[3].set_title("Residual")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Could not perform decomposition: {e}")

    def create_correlation_heatmap(self):
        """Create correlation heatmap of weather variables"""
        if len(self.weather_cols) < 2:
            print("Need at least 2 weather variables for correlation")
            return

        corr_matrix = self.df[self.weather_cols].corr()

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
        plt.title("Correlation Heatmap of Weather Variables")
        plt.tight_layout()
        plt.show()

    def plot_daily_aggregates(self, column):
        """Plot daily aggregates (mean, min, max)"""
        if column not in self.df.columns:
            print(f"Column '{column}' not found")
            return

        daily_data = self.df[column].resample("D").agg(["mean", "min", "max"])

        plt.figure(figsize=(14, 6))
        plt.plot(daily_data.index, daily_data["mean"], label="Mean", linewidth=1.5)
        plt.fill_between(
            daily_data.index,
            daily_data["min"],
            daily_data["max"],
            alpha=0.2,
            label="Min/Max Range",
        )
        plt.title(f"Daily {column} - Mean, Min, Max")
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def create_interactive_dashboard(self):
        """Create an interactive Plotly dashboard"""
        if len(self.weather_cols) < 1:
            print("No weather data available")
            return

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Temperature",
                "Humidity",
                "Correlation Matrix",
                "Distribution",
                "Monthly Average",
                "Recent Trend",
            ),
        )

        # Add time series plots
        for i, col in enumerate(self.weather_cols[:2]):
            row = i + 1
            fig.add_trace(
                go.Scatter(x=self.df.index, y=self.df[col], mode="lines", name=col),
                row=row,
                col=1,
            )

        # Add distribution plots (simplified in Plotly)
        if len(self.weather_cols) >= 1:
            fig.add_trace(
                go.Histogram(
                    x=self.df[self.weather_cols[0]], name="Distribution", nbinsx=30
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(height=800, showlegend=True, title_text="Weather Dashboard")
        fig.show()

    def plot_monthly_trends(self, column):
        """Plot monthly averages and trends"""
        if column not in self.df.columns:
            print(f"Column '{column}' not found")
            return

        monthly_avg = self.df[column].resample("M").mean()

        plt.figure(figsize=(14, 6))
        plt.plot(monthly_avg.index, monthly_avg.values, marker="o", linewidth=2)
        plt.title(f"Monthly Average {column}")
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Return monthly statistics
        return monthly_avg.describe()
