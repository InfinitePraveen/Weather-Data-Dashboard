import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import WeatherDataLoader
from visualizations import WeatherVisualizer
import warnings

warnings.filterwarnings("ignore")


class WeatherDashboard:
    def __init__(self, data_path):
        """Initialize the dashboard with data path"""
        self.loader = WeatherDataLoader(data_path)
        self.visualizer = WeatherVisualizer(self.loader)

    def run_full_analysis(self):
        """Run complete analysis and display all visualizations"""
        print("=" * 60)
        print("WEATHER DATA DASHBOARD")
        print("=" * 60)

        # 1. Data Summary
        print("\n1. DATA SUMMARY")
        print("-" * 40)
        print(f"Total records: {len(self.loader.df)}")
        print(
            f"Date range: {self.loader.df.index.min()} to {self.loader.df.index.max()}"
        )
        print(f"Weather variables: {', '.join(self.loader.weather_cols)}")

        # 2. Summary Statistics
        print("\n2. SUMMARY STATISTICS")
        print("-" * 40)
        stats = self.loader.get_summary_stats()
        print(stats)

        # 3. Visualizations
        print("\n3. GENERATING VISUALIZATIONS")
        print("-" * 40)

        # Time Series Plot
        print("Creating time series plots...")
        self.visualizer.plot_time_series()

        # Correlation Heatmap
        print("Creating correlation heatmap...")
        self.visualizer.create_correlation_heatmap()

        # Daily aggregates for first weather variable
        if len(self.loader.weather_cols) > 0:
            print(f"Creating daily aggregates for {self.loader.weather_cols[0]}...")
            self.visualizer.plot_daily_aggregates(self.loader.weather_cols[0])

            print(f"Creating monthly trends for {self.loader.weather_cols[0]}...")
            self.visualizer.plot_monthly_trends(self.loader.weather_cols[0])

            # Seasonal decomposition
            print(
                f"Creating seasonal decomposition for {self.loader.weather_cols[0]}..."
            )
            self.visualizer.plot_seasonal_decomposition(self.loader.weather_cols[0])

        # Interactive Dashboard
        print("Creating interactive dashboard...")
        self.visualizer.create_interactive_dashboard()

        print("\nAnalysis complete!")

    def get_filtered_data(self, start_date=None, end_date=None):
        """Get filtered data for a specific date range"""
        if start_date and end_date:
            mask = (self.loader.df.index >= start_date) & (
                self.loader.df.index <= end_date
            )
            return self.loader.df[mask]
        return self.loader.df


# Example usage
if __name__ == "__main__":
    # Update with your actual file path
    data_path = "data/weather_data.csv"

    dashboard = WeatherDashboard(data_path)
    dashboard.run_full_analysis()
