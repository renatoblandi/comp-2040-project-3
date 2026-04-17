# Winnipeg Naloxone Incidents Analysis

Final project for COMP-2040 Python Essentials at Red River College Polytech.

This project analyzes naloxone administration data from Winnipeg EMS (2007-2026) to understand how, when, and where opioid overdose incidents happen in the city.

---

## Project Structure

**data/** contains three files. `naloxone.csv` is the raw dataset. `naloxone_clean.csv` is the cleaned version used in all analysis. `ward_demographics.csv` is a supplementary dataset with population and income data for each of Winnipeg's 15 wards.

**exploring/** contains all four notebooks. See the Notebooks section below.

**src/** contains `helpers.py`, a module with reusable functions used across notebooks.

---

## Notebooks

**`exploring.ipynb`**: first look at the raw dataset. Checks structure, data types, missing values, and value distributions. This is where the analytical questions came from.

**`data_cleaning.ipynb`**: documents every cleaning decision made on the raw data. Parses dates, standardizes missing values, extracts new columns, and produces `naloxone_clean.csv`.

**`prediction.ipynb`**: builds a Decision Tree classifier to predict whether an incident will need multiple naloxone doses. Covers feature selection, encoding, train/test split, training, and evaluation.

**`analysis.ipynb`**: the main deliverable. Answers four analytical questions using visualizations and connects the findings to real-world context.

---

## Analytical Questions

1. How have naloxone incidents evolved over the years?
2. Are there temporal patterns in when incidents occur?
3. Which wards are most affected and why?
4. Is age or gender related to needing multiple doses?

---

## Data Sources

**`naloxone.csv`**: [Naloxone Administrations](https://data.winnipeg.ca/Fire-and-Paramedic-Service/Naloxone-Administrations/qd6b-q49i/about_data) from the City of Winnipeg Open Data portal.

**`ward_demographics.csv`**: built from the [2021 Census Ward Profiles](https://legacy.winnipeg.ca/census/2021/Wards/) published by the City of Winnipeg. Contains population, median household income (2020), and low-income rate per ward.

---

## How to Run

1. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
2. Open notebooks in order: `exploring` > `data_cleaning` > `analysis` > `prediction`
3. All notebooks read from the `data/` folder using relative paths