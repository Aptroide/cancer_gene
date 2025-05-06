import pandas as pd
import os
import numpy as np
from multiprocessing import Pool
from rich import print

def process_single_file(args):
    """Worker function to process a single CSV file"""
    csv_file, folder_path = args
    # Get study_id from filename
    study_id = csv_file.replace('_freqAltas.csv', '')
    
    # Read and process CSV
    df = pd.read_csv(os.path.join(folder_path, csv_file))
    df.set_index('Gene', inplace=True)
    
    # Select and rename frequency column
    return df[['Frecuencia (%)']].rename(columns={'Frecuencia (%)': study_id})

def load_and_combine_csv_files(folder_path, depre_file, study_name):
    """Load and combine CSV files in parallel"""
    # Get list of CSV files
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('_freqAltas.csv')]
    
    # Process files in parallel
    with Pool() as pool:
        # Create args list for worker function
        args = [(f, folder_path) for f in csv_files]
        # Map process_single_file across all files
        dataframes = pool.map(process_single_file, args)
    
    # Combine all DataFrames
    master_df = pd.DataFrame()
    for df in dataframes:
        master_df = master_df.join(df, how='outer')
    
    # Process depression file
    frecuencia_df = pd.read_csv(depre_file)
    original_df = master_df.copy()
    original_df[study_name] = np.nan
    
    # Add depression data
    for index, row in frecuencia_df.iterrows():
        gene = row['Gene']
        frecuencia = row['Frecuencia (%)']
        
        if gene in original_df.index:
            original_df.at[gene, study_name] = frecuencia
        else:
            new_row = {col: np.nan for col in original_df.columns if col != study_name}
            new_row[study_name] = frecuencia
            new_row_df = pd.DataFrame([new_row], index=[gene])
            original_df = pd.concat([original_df, new_row_df])
    
    # Save final result
    original_df.to_csv('genes_cancer_depre_Final.csv')
    return original_df