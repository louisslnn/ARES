import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os


def load_data(path_to_data_processed_folder: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load processed training data and split into features and target.
    
    Args:
        path_to_data_processed_folder: Path to folder containing processed data files
        
    Returns:
        Tuple of (X_train, y_train) where:
            - X_train: DataFrame with feature columns
            - y_train: Series with RUL target values
    """
    train = pd.read_csv(os.path.join(path_to_data_processed_folder, 'train_relevant_features.csv'))
    test = pd.read_csv(os.path.join(path_to_data_processed_folder, 'test_relevant_features.csv'))
    
    y_train = train['RUL']
    X_train = train.drop('RUL', axis=1)

    y_test = test['RUL']
    X_test = test.drop('RUL', axis=1)

    y_train = y_train.clip(upper=125)
    y_test = y_test.clip(upper=125)

    return X_train, y_train, X_test, y_test

def evaluate(y_true, y_hat, label='test'):
    """
    Evaluate model predictions using RMSE and R² score.
    
    Args:
        y_true: True target values
        y_hat: Predicted values
        label: Label for the dataset (e.g., 'train', 'test')
        
    Returns:
        Tuple of (rmse, r2_score)
    """
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_hat)
    print('{} set RMSE: {:.4f}, R²: {:.4f}'.format(label, rmse, r2))
    return rmse, r2

