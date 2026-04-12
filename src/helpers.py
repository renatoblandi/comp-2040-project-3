"""
Helper functions for Project 3

This module contains reusable functions
for data cleaning, analysis, and visualization.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    ax.annotate("Fentanyl enters the market",
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
    ax[1].bar(incidents_hour.index, incidents_hour, color="orange", edgecolor="black", width=0.8)
    ax[1].set_xticks(np.arange(0, 24, 2))

    # labeling right bar chart
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