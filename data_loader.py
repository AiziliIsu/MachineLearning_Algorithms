import streamlit as st
import numpy as np
import pandas as pd
import os

# Define the data directory path
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_dataset(dataset_type):
    """Load dataset based on type (regression or classification)"""
    if dataset_type == "regression":
        return load_regression_dataset()
    elif dataset_type == "classification":
        return load_classification_dataset()
    else:
        raise ValueError("Invalid dataset type. Choose 'regression' or 'classification'")


def load_classification_dataset():
    """Load Titanic classification dataset from pre-split CSV files"""
    try:
        # Construct the full path to the training dataset
        train_path = os.path.join(DATA_DIR, "titanic_train.csv")

        # Load the train dataset for Titanic
        df_train = pd.read_csv(train_path)

        # Drop Passenger ID column as it's not a feature
        if 'Passenger' in df_train.columns:
            df_train = df_train.drop('Passenger', axis=1)

        # If 'Survived' column exists, rename it to 'target' for consistency
        if 'Survived' in df_train.columns:
            df_train = df_train.rename(columns={'Survived': 'target'})

        # Convert 'Sex' to numeric if it's categorical
        if 'Sex' in df_train.columns and df_train['Sex'].dtype == 'object':
            df_train['Sex'] = df_train['Sex'].map({'male': 0, 'female': 1})

        return df_train, "Titanic Survival Classification"

    except FileNotFoundError:
        st.error(f"Titanic dataset file not found. Please ensure 'titanic_train.csv' is in the {DATA_DIR} directory.")
        # Return a small empty DataFrame as a fallback
        return pd.DataFrame(), "Titanic Survival Classification (Data not found)"



def load_regression_dataset():
    """Load Flight Price regression dataset from CSV"""
    try:
        # Construct the full path to the flight price dataset
        flight_path = os.path.join(DATA_DIR, "flight_price.csv")

        # Load the flight price dataset
        df = pd.read_csv(flight_path)

        # Rename 'price' column to 'target' for consistency if it exists
        if 'price' in df.columns:
            df = df.rename(columns={'price': 'target'})

        # Encode categorical features
        categorical_cols = ['airline', 'source_city', 'departure_time',
                            'stops', 'arrival_time', 'destination_city', 'class']

        # One-Hot Encoding (Preferred)
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Drop flight number (not useful for regression)
        df.drop(columns=['flight'], inplace=True, errors='ignore')

        return df, "Flight Price Prediction"

    except FileNotFoundError:
        st.error(
            f"Flight price dataset file not found. Please ensure 'flight_price.csv' is in the {DATA_DIR} directory.")
        # Return a small empty DataFrame as a fallback
        return pd.DataFrame(), "Flight Price Prediction (Data not found)"



@st.cache_data
def get_test_data_classification():
    """Load Titanic test dataset from CSV"""
    try:
        # Construct the full path to the test dataset
        test_path = os.path.join(DATA_DIR, "titanic_test.csv")

        # Load the test dataset for Titanic
        df_test = pd.read_csv(test_path)

        # Drop Passenger ID column as it's not a feature
        if 'Passenger' in df_test.columns:
            df_test = df_test.drop('Passenger', axis=1)

        # If 'Survived' column exists, rename it to 'target' for consistency
        if 'Survived' in df_test.columns:
            df_test = df_test.rename(columns={'Survived': 'target'})

        # Convert 'Sex' to numeric if it's categorical
        if 'Sex' in df_test.columns and df_test['Sex'].dtype == 'object':
            df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1})

        return df_test

    except FileNotFoundError:
        st.error(
            f"Titanic test dataset file not found. Please ensure 'titanic_test.csv' is in the {DATA_DIR} directory.")
        # Return a small empty DataFrame as a fallback
        return pd.DataFrame()