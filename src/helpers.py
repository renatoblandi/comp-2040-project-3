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