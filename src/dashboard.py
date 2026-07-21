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

    def run_full_analysis(self, location=None):
        """Run complete analysis and display all visualizations"""
        print("=" * 60)
        print("WEATHER DATA DASHBOARD")
        if location:
            print(f"Analyzing: {location}")
        else:
            print("Analyzing: All Locations")
        print("=" * 60)

        # 1. Data Summary
        print("\n1. DATA SUMMARY")
        print("-" * 40)
        print(f"Total records: {len(self.loader.df)}")
        print(
            f"Date range: {self.loader.df.index.get_level_values('Date_Time').min()} to {self.loader.df.index.get_level_values('Date_Time').max()}"
        )
        print(f"Locations: {', '.join(self.loader.locations)}")
        print(f"Weather variables: {', '.join(self.loader.weather_cols)}")

        # 2. Location Summary
        print("\n2. LOCATION SUMMARY")
        print("-" * 40)
        location_summary = self.loader.get_location_summary()
        print(location_summary)

        # 3. Summary Statistics
        print("\n3. SUMMARY STATISTICS")
        print("-" * 40)
        stats = self.loader.get_summary_stats(location)
        print(stats)

        # 4. Visualizations
        print("\n4. GENERATING VISUALIZATIONS")
        print("-" * 40)

        # Time Series Plot
        print("Creating time series plots...")
        self.visualizer.plot_time_series(location)

        # Multiple Locations Comparison (if applicable)
        if len(self.loader.locations) > 1:
            print("Creating location comparison plots...")
            for col in self.loader.weather_cols[:2]:  # Plot first 2 variables
                self.visualizer.plot_multiple_locations(col)

        # Correlation Heatmap
        print("Creating correlation heatmap...")
        self.visualizer.create_correlation_heatmap(location)

        # Daily aggregates for first weather variable
        if len(self.loader.weather_cols) > 0:
            col = self.loader.weather_cols[0]
            print(f"Creating daily aggregates for {col}...")
            self.visualizer.plot_daily_aggregates(col, location)

            print(f"Creating monthly trends for {col}...")
            self.visualizer.plot_monthly_trends(col, location)

            # Seasonal decomposition
            print(f"Creating seasonal decomposition for {col}...")
            self.visualizer.plot_seasonal_decomposition(col, location)

        # Location comparison boxplot
        if len(self.loader.locations) > 1 and len(self.loader.weather_cols) > 0:
            print("Creating location comparison boxplots...")
            self.visualizer.plot_location_comparison(self.loader.weather_cols[0])

        # Interactive Dashboard
        print("Creating interactive dashboard...")
        self.visualizer.create_interactive_dashboard(location)

        print("\nAnalysis complete!")

    def analyze_location(self, location):
        """Run analysis for a specific location"""
        if location not in self.loader.locations:
            print(
                f"Location '{location}' not found. Available: {self.loader.locations}"
            )
            return
        self.run_full_analysis(location)

    def get_filtered_data(self, location=None, start_date=None, end_date=None):
        """Get filtered data for a specific location and date range"""
        if location:
            data = self.loader.get_location_data(location)
        else:
            data = self.loader.df

        if start_date and end_date:
            # Reset index to filter by date
            if isinstance(data.index, pd.MultiIndex):
                data = data.reset_index(level="Location", drop=True)
            mask = (data.index >= start_date) & (data.index <= end_date)
            return data[mask]
        return data

    def export_location_report(self, location, filename=None):
        """Export a summary report for a specific location"""
        if location not in self.loader.locations:
            print(
                f"Location '{location}' not found. Available: {self.loader.locations}"
            )
            return

        if filename is None:
            filename = f"report_{location.replace(' ', '_')}.csv"

        data = self.loader.get_location_data(location)
        summary = data[self.loader.weather_cols].describe()
        summary.to_csv(filename)
        print(f"Report saved to {filename}")

        # Also save the full data
        data_filename = f"data_{location.replace(' ', '_')}.csv"
        data.to_csv(data_filename)
        print(f"Data saved to {data_filename}")


# Example usage
if __name__ == "__main__":
    # Update with your actual file path
    data_path = "data/weather_data.csv"

    dashboard = WeatherDashboard(data_path)

    # Run full analysis for all locations
    dashboard.run_full_analysis()

    # Or analyze a specific location
    # if dashboard.loader.locations:
    #     dashboard.analyze_location(dashboard.loader.locations[0])
