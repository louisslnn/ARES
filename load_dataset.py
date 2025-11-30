import pandas as pd
import os

def load_dataset(path_to_data_folder: str, path_to_loaded_data_folder: str) -> None:
    """
    Load raw dataset files and convert them to CSV format.
    
    Reads train_FD001.txt, test_FD001.txt, and RUL_FD001.txt from the raw data folder,
    assigns proper column names, and saves them as CSV files in the loaded data folder.
    
    Args:
        path_to_data_folder: Path to folder containing raw .txt data files
        path_to_loaded_data_folder: Path to folder where CSV files will be saved
    """
    unit_names = ['unit_number', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f's_{i}' for i in range(1, 22)]  # 21 sensors (s_1 through s_21)
    col_names = unit_names + setting_names + sensor_names

    train = pd.read_csv(os.path.join(path_to_data_folder, 'train_FD001.txt'), sep=r'\s+', header=None, names=col_names)
    test = pd.read_csv(os.path.join(path_to_data_folder, 'test_FD001.txt'), sep=r'\s+', header=None, names=col_names)
    y_test = pd.read_csv(os.path.join(path_to_data_folder, 'RUL_FD001.txt'), sep=r'\s+', header=None, names=['RUL'])

    # Create output directory if it doesn't exist
    os.makedirs(path_to_loaded_data_folder, exist_ok=True)
    
    # Save to CSV files
    train.to_csv(os.path.join(path_to_loaded_data_folder, 'train.csv'), index=False)
    test.to_csv(os.path.join(path_to_loaded_data_folder, 'test.csv'), index=False)
    y_test.to_csv(os.path.join(path_to_loaded_data_folder, 'y_test.csv'), index=False)

if __name__ == '__main__':
    load_dataset('data/raw', 'data/loaded')