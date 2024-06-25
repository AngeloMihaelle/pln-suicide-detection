import pandas as pd
from sklearn.model_selection import train_test_split

def split_corpus(csv_file, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """
    Splits a CSV file into training, validation, and testing sets.

    Parameters:
    - csv_file: str, path to the CSV file.
    - train_size: float, proportion of the dataset to include in the training set.
    - val_size: float, proportion of the dataset to include in the validation set.
    - test_size: float, proportion of the dataset to include in the testing set.
    - random_state: int, controls the shuffling applied to the data before applying the split.

    Returns:
    - None. Writes the splits to 'train_split.csv', 'validation_split.csv', and 'test_split.csv'.
    """
    
    # Check if the proportions sum up to 1
    if train_size + val_size + test_size != 1:
        raise ValueError("Train, validation, and test sizes must sum to 1.")
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Split into training and temporary sets
    X_train, X_temp = train_test_split(df, test_size=(val_size + test_size), random_state=random_state)
    
    # Calculate the proportion for validation set from the temporary set
    val_proportion = val_size / (val_size + test_size)

    # Split the temporary set into validation and testing sets
    X_val, X_test = train_test_split(X_temp, test_size=(1 - val_proportion), random_state=random_state)

    # Save the splits to new CSV files
    X_train.to_csv('train_split.csv', index=False)
    X_val.to_csv('validation_split.csv', index=False)
    X_test.to_csv('test_split.csv', index=False)

    print("Splits created and saved successfully.")

# Example usage
split_corpus('data_raw.csv')
