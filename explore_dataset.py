#!/usr/bin/env python3
import pandas as pd
import os

def load_frenk_dataset(data_path="data/FRENK-hate-en/dev.tsv"):
    """
    Load the FRENK hate speech dataset.
    
    Args:
        data_path (str): Path to the TSV file
    
    Returns:
        pandas.DataFrame: Dataset with columns ['id', 'text', 'detailed_label', 'binary_label', 'target', 'topic']
        
    Raises:
        FileNotFoundError: If the dataset file is not found
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")
    
    # Read the TSV file
    df = pd.read_csv(data_path, sep='\t', header=None)
    
    # Based on the README and file inspection, the columns are:
    # 0: id, 1: text, 2: detailed_label, 3: binary_label, 4: target, 5: topic
    column_names = ['id', 'text', 'detailed_label', 'binary_label', 'target', 'topic']
    df.columns = column_names
    
    return df

def get_row_by_index(df, row_index):
    """
    Get a specific row from the dataset by index.
    
    Args:
        df (pandas.DataFrame): The dataset
        row_index (int): Row index (0-based)
    
    Returns:
        pandas.Series: The row data
        
    Raises:
        IndexError: If row index is out of bounds
    """
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"Row index {row_index} out of bounds (0-{len(df)-1})")
    
    return df.iloc[row_index]

def explore_frenk_dataset():
    """
    Explore the FRENK hate speech dataset
    """
    try:
        df = load_frenk_dataset()
        print("Reading FRENK dataset...")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n" + "="*80)
    print("FIRST 5 RECORDS:")
    print("="*80)
    for idx, row in df.head().iterrows():
        print(f"\nRecord {idx + 1}:")
        print(f"ID: {row['id']}")
        print(f"Text: {row['text']}")
        print(f"Detailed Label: {row['detailed_label']}")
        print(f"Binary Label: {row['binary_label']}")
        print(f"Target: {row['target']}")
        print(f"Topic: {row['topic']}")
        print("-" * 40)
    
    print("\n" + "="*80)
    print("COLUMN ANALYSIS:")
    print("="*80)
    
    print(f"\nUnique values in 'detailed_label':")
    print(df['detailed_label'].value_counts())
    
    print(f"\nUnique values in 'binary_label':")
    print(df['binary_label'].value_counts())
    
    print(f"\nUnique values in 'target':")
    print(df['target'].value_counts())
    
    print(f"\nUnique values in 'topic':")
    print(df['topic'].value_counts())
    
    print(f"\nText length statistics:")
    text_lengths = df['text'].str.len()
    print(f"Min length: {text_lengths.min()}")
    print(f"Max length: {text_lengths.max()}")
    print(f"Mean length: {text_lengths.mean():.1f}")
    print(f"Median length: {text_lengths.median():.1f}")
    
    return df

if __name__ == "__main__":
    df = explore_frenk_dataset()