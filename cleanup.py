import pandas as pd
import matplotlib.pyplot as plt

COLUMNS = ["id", "url", "region", "region_url", "price", "year", "manufacturer",
"model", "condition", "cylinders", "fuel", "odometer", "title_status", 
"transmission", "vin", "drive", "size", "type", "paint_color", "image_url", 
"description", "county", "state", "lat", "long", "posting_date"]

DATASET_PATH = "vehicles.csv"

def drop_fields(vehicles):

    # Irrelevant fields
    vehicles = vehicles.drop(
        labels = [
            "id", "url", "region", "region_url", "vin", "paint_color",
            "image_url", "description", "county", "lat", "long", "posting_date"
        ],
        axis = 1
    )

    # Other fields.
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

    # Rows with null values
    vehicles = vehicles.dropna(how = "any")

    return vehicles


def reformat_strings(vehicles):

    vehicles["manufacturer"] = vehicles["manufacturer"].str.lower()
    vehicles["manufacturer"] = vehicles["manufacturer"].str.strip()
    vehicles["model"] = vehicles["model"].str.lower()
    vehicles["model"] = vehicles["model"].str.strip()
    vehicles["state"] = vehicles["state"].str.lower()
    vehicles["state"] = vehicles["state"].str.strip()

    return vehicles


def remove_outliers(vehicles):

    Q1 = vehicles["price"].quantile(0.25)
    Q3 = vehicles["price"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    vehicles = vehicles[(lower_bound < vehicles["price"]) & (vehicles["price"] < upper_bound)]

    Q1 = vehicles["year"].quantile(0.25)
    Q3 = vehicles["year"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    vehicles = vehicles[(lower_bound < vehicles["year"]) & (vehicles["year"] < upper_bound)]

    Q1 = vehicles["odometer"].quantile(0.25)
    Q3 = vehicles["odometer"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    vehicles = vehicles[(lower_bound < vehicles["odometer"]) & (vehicles["odometer"] < upper_bound)]

    return vehicles


if __name__ == "__main__":

    # Read in vehicles.csv
    vehicles = pd.read_csv(
        filepath_or_buffer = DATASET_PATH,
        header = 0, # Need this to replace existing column names
        names = COLUMNS,
    )

    # Drop unneeded fields
    vehicles = drop_fields(vehicles)

    # Convert to more appropriate types
    vehicles = vehicles.convert_dtypes()

    # Clean up formatting for strings
    vehicles = reformat_strings(vehicles)

    vehicles = remove_outliers(vehicles)

    # Convert to new csv
    vehicles.to_csv("new_vehicles.csv", index = False)
