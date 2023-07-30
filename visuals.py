import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats
import os

IMAGE_DIR = "images"
DATA_DIR = "new_vehicles.csv"

def print_stats(data, label: str):
    print(f"\n### {label} Statistical Data ###")
    print(f"Mean: {stats.fmean(data):.2f}")
    print(f"Median: {stats.median(data):.2f}")
    print(f"Quantiles: {stats.quantiles(data)}")
    plt.clf()
    plt.boxplot(data)
    plt.savefig(f"images/{label}_boxplot.png")


if __name__ == "__main__":

    # Make the directory to store images
    os.makedirs(IMAGE_DIR, exist_ok = True)

    # Read in dataset
    vehicles = pd.read_csv(DATA_DIR)

    # Convert to more appropriate types
    vehicles = vehicles.convert_dtypes()

    print_stats(vehicles["price"], "Price")
    print_stats(vehicles["year"], "Year")
    print_stats(vehicles["odometer"], "Odometer")