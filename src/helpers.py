"""
Helper functions for Project 3

This module contains reusable functions
for data cleaning, analysis, and visualization.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.preprocessing import OneHotEncoder

def expand_dispatch_date(df):
    """Parse dispatch_date and expand it into separate temporal columns.

    Converts the dispatch_date string column to datetime, extracts year,
    month, day_of_week, hour, and date into individual columns, then drops
    the original dispatch_date column.
    
    Parameters:
        df: DataFrame containing a dispatch_date column in datetime ISO format.
    
    Returns:
        DataFrame with dispatch_date replaced by the extracted columns.
    """

    df = df.copy() # don't modifies the original

    # parse to datetime
    df["dispatch_date"] = pd.to_datetime(df["dispatch_date"])

    # extract individual columns
    df["date"] = df["dispatch_date"].dt.date
    df["year"] = df["dispatch_date"].dt.year
    df["month"] = df["dispatch_date"].dt.month
    df["day_of_week"] = df["dispatch_date"].dt.day_of_week
    df["hour"] = df["dispatch_date"].dt.hour

    # drop now redundant, dispatch_date column
    df = df.drop(columns=["dispatch_date"])

    return df

def age_range_to_midpoint(age_range):
    """Convert a string age range to its midpoint as a number.
    
    Parameters:
        age_range: A string like "25 to 29", or NaN if missing

    Returns:
        The midpoint as a float, 100 for "Over 100",
        or NaN if it is a missing value.
    """
    if pd.isna(age_range):
        return np.nan

    parts = age_range.split(" ")
    lower = parts[0]
    upper = parts[-1]

    if lower == "Over": # handles the "Over 100" edge case
        return 100
    
    midpoint = (int(lower) + int(upper)) / 2

    return midpoint

def reorder_columns(df):
    """Reorder dataset columns to a more readable order.

    Parameters:
        df: Transformed DataFrame with new columns.

    Returns:
        A nicely ordered DataFrame.
    """

    df = df.copy()

    new_order = [
    "incident_number",
    "patient_number",
    "date",
    "year",
    "month",
    "day_of_week",
    "hour",
    "age",
    "age_midpoint",
    "gender",
    "ward",
    "naloxone_administrations",
    "is_multiple_dose"
    ]

    df = df[new_order]

    return df

def merge_ward_data(df, wards_df):
    """Aggregate incidents by ward and merge with demographic data.

    Parameters:
        df: The cleaned naloxone DataFrame.
        wards_df: The ward demographics DataFrame.

    Returns:
        A DataFrame with one row per ward, containing total incidents,
        population, median household income, and incidents per 1,000 residents.
    """

    # reset index to turn groupby into a regular df
    by_ward = df.groupby("ward")["incident_number"].nunique().reset_index()
    by_ward.columns = ["ward", "incidents"] # renaming incident_number to incidents
    by_ward = by_ward.merge(wards_df, on="ward") # merging supplementary dataset on `ward` column

    # creating new column to hold incidents per 1000 residents
    by_ward["incidents_per_1000"] = (by_ward["incidents"] / by_ward["population"]) * 1000

    return by_ward

def plot_incidents_by_year(df):
    """Plot total incidents and naloxone administrations by year.

    Parameters:
        df: The cleaned naloxone DataFrame.
    """
    
    # preparing the data to plot
    by_year = df.groupby("year")
    administrations_year = by_year["naloxone_administrations"].sum().loc[2008:2025]
    incidents_year = by_year["incident_number"].nunique().loc[2008:2025]

    # plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # plot both lines
    ax.plot(administrations_year.index, administrations_year, color="red", label="Administrations")
    ax.plot(incidents_year.index, incidents_year, color="orange", label="Incidents")

    # fill both areas
    ax.fill_between(incidents_year.index, incidents_year, color="orange", alpha=0.3)
    ax.fill_between(administrations_year.index, incidents_year, administrations_year, color="red", alpha=0.3)

    # text + arrow annotation about fentanyl
    ax.annotate("Fentanyl crisis start",
                xy=(2015, administrations_year[2015]),
                xytext=(2015 - 5, administrations_year[2015] + 1000),
                arrowprops=dict(arrowstyle="fancy", color="black"),
                fontsize=13)


    ax.set_xticks([2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024])

    # labeling the plot
    ax.set_title("Number of Incidents and Administrations by Year", fontsize=16, fontweight="bold")
    ax.set_xlabel("Year", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.5)

    # displaying the chart
    ax.legend()
    plt.show()

def plot_temporal_patterns(df):
    """Plot incidents by day of week and by hour of day.

    Parameters:
        df: The cleaned naloxone DataFrame.
    """
    
    # preparing the data
    incidents_dow = df.groupby("day_of_week")["incident_number"].nunique()
    incidents_hour = df.groupby("hour")["incident_number"].nunique()

    hours_reordered = list(range(6, 24)) + list(range(0, 6))
    incidents_hour = incidents_hour.reindex(hours_reordered)

    # plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # plotting left bar chart (incidents by day of week)
    ax[0].bar(incidents_dow.index, incidents_dow, color="orange", edgecolor="black")
    ax[0].set_xticks(np.arange(0, 7))

    # labeling left bar chart
    ax[0].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax[0].set_title("Incidents by Day of Week", fontsize=16, fontweight="bold")
    ax[0].set_xlabel("Day of Week", fontsize=13, fontweight="bold")
    ax[0].set_ylabel("Number of Incidents", fontsize=13, fontweight="bold")

    # plotting right bar chart (incidents by hour)
    ax[1].bar(range(24), incidents_hour, color="orange", edgecolor="black", width=0.8)
    ax[1].set_xticks(range(0, 24, 2))

    # labeling right bar chart
    ax[1].set_xticklabels([f"{h}h" for h in hours_reordered[::2]])
    ax[1].set_title("Incidents by Hour", fontsize=16, fontweight="bold")
    ax[1].set_xlabel("Hour", fontsize=13, fontweight="bold")
    ax[1].set_ylabel("Number of Incidents", fontsize=13, fontweight="bold")

    # setting the grid for both charts
    ax[0].grid(axis="y", alpha=0.5)
    ax[1].grid(axis="y", alpha=0.5)

    # displaying the figure
    plt.show()

def plot_ward_incidents(df, wards_df):
    """Plot total incidents and incidents per 1,000 residents by ward.

    Parameters:
        df: The cleaned naloxone DataFrame.
        wards_df: The ward demographics DataFrame. 
    """

    # preparing the data
    by_ward = merge_ward_data(df, wards_df)

    # creating new column to hold incidents per 1000 residents
    by_ward["incidents_per_1000"] = (by_ward["incidents"] / by_ward["population"]) * 1000

    sorted_df = by_ward.sort_values("incidents")
    sorted_normalized_df = by_ward.sort_values("incidents_per_1000")

    # plotting
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # plotting left barh chart (not normalized - raw total incidents)
    ax[0].barh(sorted_df["ward"], sorted_df["incidents"], color="orange", edgecolor="black")

    # labeling left barh
    ax[0].set_title("Incidents by Ward (Total)", fontsize=16, fontweight="bold")
    ax[0].set_xlabel("Number of Incidents", fontsize=13, fontweight="bold")
    ax[0].set_ylabel("Ward", fontsize=13, fontweight="bold")
    ax[0].grid(axis="x", alpha=0.5)

    # plotting right barh (normalized by incidents per 1000 residents)
    ax[1].barh(sorted_normalized_df["ward"], sorted_normalized_df["incidents_per_1000"],
            color="orange", edgecolor="black")

    # labeling right barh
    ax[1].set_title("Incidents by Ward (per 1,000 residents)", fontsize=16, fontweight="bold")
    ax[1].set_xlabel("Incidents per 1,000 Residents", fontsize=13, fontweight="bold")
    ax[1].set_ylabel("Ward", fontsize=13, fontweight="bold")
    ax[1].grid(axis="x", alpha=0.5)

    # displaying the figure
    plt.tight_layout()
    plt.show()

def plot_income_scatter(df, wards_df):
    """Plot median household income vs incidents per 1,000 residents by ward.

    Parameters:
        df: The cleaned naloxone DataFrame.
        wards_df: The ward demographics DataFrame.
    """

    # preparing the data
    by_ward = merge_ward_data(df, wards_df)

    # creating new column to hold incidents per 1000 residents
    by_ward["incidents_per_1000"] = (by_ward["incidents"] / by_ward["population"]) * 1000

    # all wards except Point Douglas
    other_wards = by_ward[by_ward["ward"] != "Point Douglas"]
    point_douglas = by_ward[by_ward["ward"] == "Point Douglas"]

    # plotting
    fig, ax = plt.subplots(figsize=(10, 7))

    # plotting "other wards" scatter
    ax.scatter(other_wards["median_household_income"], other_wards["incidents_per_1000"],
            s=120, color="steelblue", edgecolor="black", alpha=0.85)

    # plotting "point douglas" scatter (just one point with different color)
    ax.scatter(point_douglas["median_household_income"], point_douglas["incidents_per_1000"],
            s=120, color="red", edgecolor="black", alpha=0.85)

    # quadratic trend line
    coeffs = np.polyfit(by_ward["median_household_income"], by_ward["incidents_per_1000"], 2)
    x_line = np.linspace(by_ward["median_household_income"].min(), by_ward["median_household_income"].max(), 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), color="red", linestyle="--", alpha=0.5, label="Quadratic best-fit")

    # annotating the main wards
    main_wards = ["Mynarski", "Daniel McIntyre", "Fort Rouge - East Fort Garry"]
    main_wards_df = by_ward[by_ward["ward"].isin(main_wards)]
    for _, row in main_wards_df.iterrows():
        ax.annotate(row["ward"], (row["median_household_income"], row["incidents_per_1000"]),
                    fontsize=8, xytext=(5, 4), textcoords="offset points")

    # Point Douglas annotation
    ax.annotate("Point Douglas\n(concentrates users from\nacross the city)",
                (point_douglas["median_household_income"].values[0], point_douglas["incidents_per_1000"].values[0]),
                fontsize=8, xytext=(5, -45), textcoords="offset points")

    # labeling the plot
    ax.set_title("Median Household Income vs Incidents per 1,000 Residents", fontsize=16, fontweight="bold")
    ax.set_xlabel("Median Household Income (2020)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Incidents per 1,000 Residents", fontsize=13, fontweight="bold")

    # displaying the chart
    ax.grid(alpha=0.4)
    plt.legend()
    plt.show()

def plot_multiple_dose_heatmap(df):
    """Plot multiple dose rate by gender and age group as a heatmap.

    Parameters:
        df: The cleaned naloxone DataFrame.
    """

    # preparing the data

    # separating age range in bigger sections (age_group)
    bins = [0, 19, 29, 39, 49, 59, 200]
    labels = ["Under 20", "20s", "30s", "40s", "50s", "60+"]
    df["age_group"] = pd.cut(df["age_midpoint"], bins=bins, labels=labels)

    pivot = df.groupby(["gender", "age_group"], observed="True")["is_multiple_dose"].mean() * 100
    pivot = pivot.unstack()

    # plotting
    fig, ax = plt.subplots(figsize=(10, 4))

    # creating a heatmap for % of multiple dose by gender (y-axis) and age_group (x_axis)
    # showing the percentages for each point (square)
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5,
                cbar_kws={"label": "% Multiple Dose"}, ax=ax)

    # labeling the plot
    ax.set_title("Multiple Dose Rate (%) by Gender and Age Group", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age Group", fontsize=11)
    ax.set_ylabel("Gender", fontsize=11)

    # displaying the chart
    plt.tight_layout()
    plt.show()

def prepare_data_for_train(df):
    """
    Selects features, handles missing values, and splits/encodes the data for training.

    Categorical columns (age, gender, ward) are one-hot encoded using a encoder
    fitted only on the training set to avoid data leakage.

    Parameters:
    df : pd.DataFrame
        The cleaned naloxone dataset. Must contain: year, month, day_of_week,
        hour, age, gender, ward, is_multiple_dose.

    Returns:
    X_train, X_test : pd.DataFrame
        Feature matrices for training (80%) and testing (20%).
    y_train, y_test : pd.Series
        Target labels (is_multiple_dose) for each split.
    """

    df = df.copy() # don't modify the original

    # select the features
    selected_columns = [
    "year",
    "month",
    "day_of_week",
    "hour",
    "age",
    "gender",
    "ward",
    "is_multiple_dose"
    ]

    df = df[selected_columns]

    # substitute missing values
    df = df.fillna("Unknown")

    # split into features / target
    X = df.drop(columns=["is_multiple_dose"]) # features
    y = df["is_multiple_dose"] # target

    # split into train / test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.2, stratify=y, random_state=42
    )

    # encoding
    cat_cols = ["age", "gender", "ward"]

    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    oh.fit(X_train[cat_cols])

    train_enc = oh.transform(X_train[cat_cols])
    test_enc  = oh.transform(X_test[cat_cols])

    feature_names = oh.get_feature_names_out(cat_cols)

    train_enc = pd.DataFrame(train_enc, columns=feature_names, index=X_train.index)
    test_enc  = pd.DataFrame(test_enc, columns=feature_names, index=X_test.index)

    num_cols = ["year", "month", "day_of_week", "hour"]

    X_train_final = pd.concat([X_train[num_cols], train_enc], axis=1)
    X_test_final  = pd.concat([X_test[num_cols], test_enc], axis=1)

    return X_train_final, X_test_final, y_train, y_test