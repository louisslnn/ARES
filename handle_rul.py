import pandas as pd
import os

sensor_names = [f's_{i}' for i in range(1, 22)]  # 21 sensors (s_1 through s_21)


def load_RUL(path_to_RUL_file: str) -> None:
    """
    Load RUL data from a text file and save it as a CSV file.
    
    Args:
        path_to_RUL_file: Name of the RUL file in data/raw/ directory
    """
    with open(os.path.join('data', 'raw', path_to_RUL_file), 'r') as file:
        RUL = file.readlines()
    RUL = [int(line.strip()) for line in RUL]

    os.makedirs(os.path.join('data', 'rul'), exist_ok=True)
    pd.DataFrame(RUL, columns=['RUL']).to_csv(os.path.join('data', 'rul', path_to_RUL_file[:10] + '.csv'), index=False)


def add_remaining_useful_life(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and add Remaining Useful Life (RUL) column to the dataframe.
    
    RUL is calculated as: max_cycle_per_unit - current_cycle
    
    Args:
        df: DataFrame with 'unit_number' and 'time_cycles' columns
        
    Returns:
        DataFrame with added 'RUL' column
    """
    grouped_by_unit = df.groupby(by='unit_number')
    max_cycle = grouped_by_unit['time_cycles'].max()
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_number', right_index=True)

    remaining_useful_life = result_frame['max_cycle'] - result_frame['time_cycles']
    result_frame['RUL'] = remaining_useful_life

    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame

if __name__ == '__main__':
    df_train = pd.read_csv(os.path.join('data', 'loaded', 'train.csv'))
    df_train = add_remaining_useful_life(df_train)
    df_train[sensor_names+['RUL']].head()

    for file in os.listdir(os.path.join('data', 'raw')):
        if file.startswith('RUL'):
            load_RUL(file)