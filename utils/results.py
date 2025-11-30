"""
Results tracking utility for model evaluation metrics.

This module provides functions to save and update model results in a centralized results.csv file.
"""
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Optional


RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'results.csv')


def save_results(
    model_name: str,
    rmse_train: float,
    r2_train: float,
    rmse_test: float,
    r2_test: float,
    timestamp: Optional[str] = None,
    additional_metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Save or update model results in the centralized results.csv file.
    
    If the model already exists, it will be updated. Otherwise, a new row will be added.
    
    Args:
        model_name: Name of the model (e.g., 'LinearRegression', 'RandomForest', etc.)
        rmse_train: RMSE on training set
        r2_train: R² score on training set
        rmse_test: RMSE on test set
        r2_test: R² score on test set
        timestamp: Optional timestamp string. If None, current timestamp is used.
        additional_metrics: Optional dictionary of additional metrics to include
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.dirname(RESULTS_FILE)
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare the new row
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    new_row = {
        'Model': model_name,
        'RMSE-Train': rmse_train,
        'R2-Train': r2_train,
        'RMSE-Test': rmse_test,
        'R2-Test': r2_test,
        'Timestamp': timestamp
    }
    
    # Add additional metrics if provided
    if additional_metrics:
        new_row.update(additional_metrics)
    
    # Load existing results or create new DataFrame
    if os.path.exists(RESULTS_FILE):
        results_df = pd.read_csv(RESULTS_FILE)
        
        # Check if model already exists
        if model_name in results_df['Model'].values:
            # Update existing row
            model_idx = results_df[results_df['Model'] == model_name].index[0]
            for key, value in new_row.items():
                if key in results_df.columns:
                    results_df.at[model_idx, key] = value
                else:
                    results_df[key] = None
                    results_df.at[model_idx, key] = value
        else:
            # Add new row
            # Ensure all columns exist
            for key in new_row.keys():
                if key not in results_df.columns:
                    results_df[key] = None
            
            # Create a row with all columns
            row_dict = {col: None for col in results_df.columns}
            row_dict.update(new_row)
            results_df = pd.concat([results_df, pd.DataFrame([row_dict])], ignore_index=True)
    else:
        # Create new DataFrame
        results_df = pd.DataFrame([new_row])
    
    # Save to CSV
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved for {model_name} in {RESULTS_FILE}")


def get_results(model_name: Optional[str] = None) -> pd.DataFrame:
    """
    Get results from the results.csv file.
    
    Args:
        model_name: Optional model name to filter results. If None, returns all results.
        
    Returns:
        DataFrame with results. Empty DataFrame if file doesn't exist.
    """
    if not os.path.exists(RESULTS_FILE):
        return pd.DataFrame()
    
    results_df = pd.read_csv(RESULTS_FILE)
    
    if model_name:
        results_df = results_df[results_df['Model'] == model_name]
    
    return results_df


def print_results_summary() -> None:
    """Print a formatted summary of all model results."""
    results_df = get_results()
    
    if results_df.empty:
        print("No results found.")
        return
    
    print("\n" + "="*80)
    print("MODEL RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    print_results_summary()
