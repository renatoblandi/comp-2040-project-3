"""
Helper functions for Project 3

This module contains reusable functions
for data cleaning, analysis, and visualization.
"""
import pandas as pd
import numpy as np

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