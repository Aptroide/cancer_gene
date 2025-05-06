import os
import pandas as pd
from rich import print

def validate_csv_file(file_path):
    """
    Validates a CSV file ensuring it meets certain restrictions:
    1. The file name must follow a specific pattern.
    2. The file must contain the columns 'Gene' and 'Frequency (%)'.
    3. The data types must be correct.
    4. The 'Gene' column must have unique values.
    5. The 'Frequency (%)' column must be in the range [0, 100] and have no null values.

    Parameters:
    file_path (str): Full path of the CSV file to validate.

    Returns:
    bool: True if the file meets all restrictions, False otherwise.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return False

    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return False

    # Check that the required columns are present
    required_columns = df.columns[:2]  # Take the first two columns
    if len(required_columns) < 2:
        print("Error: The file must have at least two columns.")
        return False

    # Check data types
    for index, value in df.iloc[:, 0].items():
        try:
            float(value)
            print(f"Error: The value in row {index + 1} of the first column can be converted to 'float'.")
            return False
        except ValueError:
            continue

    if not pd.api.types.is_float_dtype(df.iloc[:, 1]):
        print("Error: The second column must contain 'float' values.")
        return False

    # Check that the 'Gene' column is unique
    if df['Gene'].duplicated().any():
        print("Error: The 'Gene' column contains duplicate values.")
        return False

    # Check that 'Frequency (%)' is numeric and in the range [0, 1]
    if not df.iloc[:, 1].between(0, 1).all():
        print("Error: The values in 'Frequency (%)' must be in the range 0 to 100.")
        return False

    # Check that there are no null values in the essential columns
    if df[required_columns].isnull().any().any():
        print("Error: There must be no null values in the 'Gene' or 'Frequency (%)' columns.")
        return False

    return True