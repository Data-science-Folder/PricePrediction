import pandas as pd


def read_data(filepath, date_col_name):
    """
    Read csv with sales data
    :return: pd.DataFrame

    """
    dataset = pd.read_csv(filepath, index_col=0, parse_dates=True)
    dataset[date_col_name] = pd.to_datetime(dataset[date_col_name])  # Convert to datetime format
    dataset.set_index(date_col_name, inplace=True)
    return dataset
