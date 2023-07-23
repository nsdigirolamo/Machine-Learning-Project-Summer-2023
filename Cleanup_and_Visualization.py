import pandas as pd
import matplotlib.pyplot as plt

def cleanup():

    """
    # Read in vehicles.csv
    vehicles: DataFrame = pd.read_csv("vehicles.csv")

    # Drop unneeded fields
    vehicles = drop_fields(vehicles)

    # Convert to more appropriate types
    vehicles = vehicles.convert_dtypes()

    # Clean up formatting for strings
    vehicles["manufacturer"] = vehicles["manufacturer"].str.lower()
    vehicles["manufacturer"] = vehicles["manufacturer"].str.strip()
    vehicles["model"] = vehicles["model"].str.lower()
    vehicles["model"] = vehicles["model"].str.strip()
    vehicles["state"] = vehicles["state"].str.lower()
    vehicles["state"] = vehicles["state"].str.strip()

    # Convert to new csv
    vehicles.to_csv("new_vehicles.csv", index = False)
    """

    # Read in new_vehicles.csv
    vehicles: DataFrame = pd.read_csv("new_vehicles.csv")
    # Convert to more appropriate types
    vehicles = vehicles.convert_dtypes()

    print("\n### Vehicles Data Types ###")
    print(vehicles.dtypes)
    print("\n### Vehicles Description Table ###")
    print(vehicles.describe())
    print("\n### Manufacturer Counts ###")
    show_manufacturer_counts(vehicles)
    print("\n### Year Counts ###")
    show_year_counts(vehicles)


def show_manufacturer_counts(vehicles):
    manufacturer_counts = vehicles["manufacturer"].value_counts()
    print(manufacturer_counts)
    manufacturer_counts.plot(
        kind = "barh",  
        figsize = (11, 8.5),
        title = "Count per Manufacturer",
        xlabel = "Count",
        ylabel = "Manufacturer",
        fontsize = 10
    )
    plt.savefig("images/manufacturer_counts.png")


def show_year_counts(vehicles):
    year_counts = vehicles["year"].value_counts()
    print(year_counts)
    year_counts.plot(
        kind = "barh",  
        figsize = (11, 8.5),
        title = "Count per Year",
        xlabel = "Count",
        ylabel = "Year",
        fontsize = 10
    )
    plt.savefig("images/year_counts.png")


def drop_fields(vehicles):

    # Drop irrelevant fields
    vehicles = vehicles.drop(
        labels = [
            "id", "url", "region", "region_url", "VIN", "paint_color",
            "image_url", "description", "county", "lat", "long", "posting_date"
        ],
        axis = 1
    )

    # Drop other fields.
    """
        condition: Too subjective.
        cylinders: May be useful, but has a lot of blanks.
        fuel: May be useful, but has a lot of blanks and "other" values.
        title_status: Most titles seem to be "clean" or blank.
        transmission: A lot of "other" values. What kind of transmission is there besides manual or automatic?
        drive: May be useful, but has a lot of blanks.
        size: A lot of blanks.
        type: May be useful, but has a lot of blanks.
    """
    vehicles = vehicles.drop(
        labels = [
            "condition", "cylinders", "fuel", "title_status", "transmission", 
            "drive", "size", "type"
        ],
        axis = 1
    )

    return vehicles


if __name__ == "__main__":
    cleanup()
