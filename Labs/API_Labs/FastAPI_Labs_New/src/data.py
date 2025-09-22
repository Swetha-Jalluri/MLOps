import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the WineQT dataset and return the features and target values.
    Returns:
        X (numpy.ndarray): The features of the WineQT dataset.
        y (numpy.ndarray): The target values of the WineQT dataset.
    """
    # Load WineQT dataset
    df = pd.read_csv("../WineQT.csv")
    
    # Drop Id column and quality (target)
    X = df.drop(['Id', 'quality'], axis=1).values
    y = df['quality'].values
    
    return X, y

def split_data(X, y):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    return X_train, X_test, y_train, y_test