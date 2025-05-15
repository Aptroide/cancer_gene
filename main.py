import os
import sys

from modules import data_extration, val_csv, dim_red, k_means, dbscan
from modules import proc_data
from modules.classification import run_pipeline
from bravado.client import SwaggerClient
import pandas as pd
import time
import json
from multiprocessing import Pool
from typing import List, Tuple, Dict
from rich import print

def procesar_dataframe(data_filtered):
    # Create a copy of the original DataFrame
    df_filtered = data_filtered.copy()
    # Reset the index to ensure it starts from 0
    df_filtered = df_filtered.reset_index()
    # Rename the column that contains the genes to "Gene"
    df_filtered = df_filtered.rename(columns={"index": "Gene"})
    # Ensure the index is numeric again
    df_filtered = df_filtered.reset_index(drop=True)
    # Fill NaN values with 0.0
    df_filtered.fillna(0.0, inplace=True)
    # Transpose the DataFrame
    df_transposed = df_filtered.transpose()

    # Use the first row as the header
    df_transposed.columns = df_transposed.iloc[0]  # Set the first row as the header
    df_transposed = df_transposed[1:]  # Remove the first row, as it is now the header

    # Reset the index and rename it to 'Cancer'
    df_transposed.reset_index(inplace=True)
    df_transposed = df_transposed.rename(columns={'index': 'Cancer'})

    # Remove the index name
    df_transposed.index.name = None

    # Convert all columns (except 'Cancer') to float64 type
    for column in df_transposed.columns[1:]:
        df_transposed[column] = pd.to_numeric(df_transposed[column], errors='coerce')

    return df_transposed

def initialize_cbioportal():
    """
    Initialize and configure the cBioPortal client.
    """
    cbioportal = SwaggerClient.from_url(
        'https://www.cbioportal.org/api/v2/api-docs',
        config={
            "validate_requests": False,
            "validate_responses": False,
            "validate_swagger_spec": False,
            "use_models": False,
        }
    )

    # Normalize method names
    for a in dir(cbioportal):
        if not a.startswith('_'):
            cbioportal.__setattr__(a.replace(' ', '_').lower(), cbioportal.__getattr__(a))

    return cbioportal

cbioportal = initialize_cbioportal()

def safe_filter_dataframe(df, genes_list):
    """
    Safely filter DataFrame by genes, handling missing ones
    """
    # Convert all genes to uppercase for consistency
    genes_list = [gene.upper() for gene in genes_list]
    
    # Find which genes exist in the DataFrame
    existing_genes = [gene for gene in genes_list if gene in df.index]
    missing_genes = set(genes_list) - set(existing_genes)
    
    if missing_genes:
        print(f"Warning: Following genes not found in cBioportal data: {missing_genes}")
    
    # Filter only existing genes
    return df.loc[existing_genes]

def study_file_exists(study_id: str) -> bool:
    """Check if study file already exists"""
    filename = f"Frequencies/{study_id}_freqAltas.csv"
    return os.path.exists(filename)

def process_single_study(study_id: str) -> Tuple[str, bool]:
    """Process single study if not already processed"""
    try:
        if study_file_exists(study_id):
            # print(f"Study {study_id} already processed, skipping...")
            return (study_id, True)
            
        data_extration.process_study(cbioportal, study_id, config['num_genes_per_study'])
        return (study_id, True)
    except Exception as e:
        # print(f"An error occurred while processing the study '{study_id}': {e}")
        return (study_id, False)

def process_studies(study_ids: List[str]) -> Tuple[int, List[str]]:
    """Process studies in parallel and handle failures"""
    limited_study_ids = study_ids[:config['num_cancer_studies']]
    with Pool() as pool:
        results = pool.map(process_single_study, limited_study_ids)
    
    successful = sum(1 for _, success in results if success)
    failed_studies = [study_id for study_id, success in results if not success]
    return successful, failed_studies

def process_failed_studies(failed_studies: List[str]):
    print(f"\nProcessing {len(failed_studies)} failed studies...")
    aux = 0
    aux2 = 0
    for study_id in failed_studies:
        aux += 1
        if aux == config['num_cancer_studies']:
            break
        try:
            data_extration.process_study(cbioportal, study_id, config['num_genes_per_study'])   
            aux2 += 1    
        except Exception as e:
            print(f"An error occurred while processing the study '{study_id}': {e}")
            continue  # Continue with the next study in case of error
    return aux2   

def load_study_ids(output_path: str) -> List[str]:
    """Load study IDs from file"""
    with open(output_path, 'r') as file:
        return [line.strip() for line in file.readlines()]
    

def process_data_for_cluster(file_path: str, freq_folder: str = 'Frequencies') -> Dict[str, float]:
    """
    Process data files and run clustering analysis
    
    Args:
        file_path: Path to depression genes file
        freq_folder: Folder containing frequency files
        
    Returns:
        Dictionary with execution times for each step
    """
    times = {}
    all_results = {}
    # Load depression genes
    start_time = time.time()
    depresion = pd.read_csv(file_path)
    genes_depresion = depresion['Gene'].tolist()
    print_fancy_title("Combining frequency files")
    # Process and combine files
    master_df = proc_data.load_and_combine_csv_files(freq_folder, file_path, config['your_study_name'])
    print(f"Total studies with at least one mutation: {master_df.shape}")

    # Filter and prepare data
    filtered_df = safe_filter_dataframe(master_df, genes_depresion)
    filtered_df = filtered_df.dropna(axis=1, how='all')
    filtered_df.to_csv(f'./{config["output_path"]}/ClustersCSV/gene_comparison.csv')
    
    print(f"Filtered data has dimensions '{filtered_df.shape}'.")
    times['data_processing'] = time.time() - start_time

    # Dimension reduction
    print_fancy_title("-------Starting Dimension Reduction-------")
    df_transposed = procesar_dataframe(filtered_df)
    features = df_transposed.iloc[:, 1:]
    start_time = time.time()
    proj2d, kpca_features, tsne_features = dim_red.dim_reduction(df_transposed, features)
    times['dim_reduction'] = time.time() - start_time
    print("Pre-processing complete.\nDimensions done: umap, kpca, and tsne")
    
    # Run clustering for each method
    for method, data in [
        ('umap', proj2d),
        ('kpca', kpca_features),
        ('tsne', tsne_features)
    ]:
        # Run clustering for each method
        text = f"Processing {method.upper()}"
        print_fancy_title(text)
        method_times, method_results = run_clustering_for_method(method, data, df_transposed)
        times.update(method_times)
        all_results.update(method_results)

    best_key = max(all_results.keys(), key=lambda k: all_results[k]['score'])
    best_result = all_results[best_key]
    
    return times, best_result['algorithm'], best_result['method']

def print_fancy_title(method: str):
    title = f"{method.upper()} "
    box_width = len(title) + 4  # Para ajustar el ancho del cuadro

    print("\n" + "═" * box_width)
    print(f"║{title.center(box_width - 2)}║")
    print("═" * box_width)

def run_clustering_for_method(method: str, data, df_transposed) -> Dict[str, float]:
    """Run K-means and DBSCAN for a given dimension reduction method"""
    times = {}
    results = {}
    
    # K-means
    print("Starting K-Means...")
    k_kmeans = k_means.c_kmeans(data)
    start_time = time.time()
    _, _, labels, kmeans_score = k_means.plot_kmeans_silhouette_plotly(
        data, 
        df_transposed, 
        [k_kmeans],
        method,
        config["output_path"],
    )

    # Save the labels to a CSV file
    labels.to_csv(f'./{config["output_path"]}/ClustersCSV/K-means_labels_{method}.csv', index=False)
    print(f"The value of the 'Cluster' column for {config['your_study_name']} in K-Means {method.upper()} is: {labels['Cluster'].iloc[-1]}")
    times[f'kmeans_{method}'] = time.time() - start_time
    
    # Store K-means results for this method
    key = f"kmeans_{method}"
    results[key] = {
        'score': kmeans_score,
        'labels': labels,
        'algorithm': 'K-means',
        'method': method
    }
    
    # DBSCAN
    print("\nStarting DBSCAN...")
    start_time = time.time()
    _, labels, dbscan_score = dbscan.optimize_dbscan(data, df_transposed, method, config["output_path"])
    # Save the labels to a CSV file
    labels.to_csv(f'./{config["output_path"]}/ClustersCSV/DBSCAN_labels_{method}.csv', index=False)
    print(f"The value of the 'Cluster' column for {config['your_study_name']} in DBSCAN {method.upper()} is: {labels['Cluster'].iloc[-1]}")
    times[f'dbscan_{method}'] = time.time() - start_time
 
    # Store DBSCAN results for this method
    key = f"dbscan_{method}"
    results[key] = {
        'score': dbscan_score,
        'labels': labels,
        'algorithm': 'DBSCAN',
        'method': method
    }
    
    return times, results


# Press the green button in the gutter to run the script.
def main():
    # Setup
    os.makedirs('Frequencies', exist_ok=True)
    # Create a subfolder within the output_path directory
    images_subfolder = os.path.join(config["output_path"], 'Figures')
    os.makedirs(images_subfolder, exist_ok=True)
    images_subfolder = os.path.join(config["output_path"], 'ClustersCSV')
    os.makedirs(images_subfolder, exist_ok=True)
    print_fancy_title("-------Starting Pre-Process-------")
    if config['studies_path'] == None:
        print("No studies_path path provided. Using all studies in Cbioportal.")
        study_ids = data_extration.fetch_all_studies(cbioportal)
        print(f"Total studies to process: {len(study_ids)}\n")
    else:
        # Load and process studies
        study_ids = load_study_ids(config['studies_path'])
        # Limit the number of studies to process
        study_ids = study_ids[:config['num_cancer_studies']]
        print(f"Total studies to process: {len(study_ids)}\n")


    # Bring the data from the cBioportal studies
    start_time = time.time()
    partial_start_time = time.time()
    successful, failed_studies = process_studies(study_ids)

    # Process failed studies
    successful += process_failed_studies(failed_studies)
    times = {'data_extraction_time': time.time() - start_time}
    
    partial_end_time = time.time()
    data_extration_time = partial_end_time - partial_start_time
    data_p = data_extration_time / 60
    print(f"Processing of {successful} studies completed in {data_p:.2f} minutes.")

    print_fancy_title("Validating CSV file")
    # Validate and process data
    if not val_csv.validate_csv_file(config['file_path']):
        print("The file does not meet the restrictions.")
        return    
    print("File is correct.")

    # TODO
    cluster_times, best_cluster, best_method = process_data_for_cluster(config['file_path'])
    times.update(cluster_times)

    # Search for files containing the word 'best_cluster' in the specified directory
    cluster_csv_path = os.path.join(config['output_path'], 'ClustersCSV')
    best_cluster_files = [
        file for file in os.listdir(cluster_csv_path)
        if best_method in file
    ]

    # Open one of the files in best_cluster_files and print the last 'Cluster' value
    if best_cluster_files and config["classification"]:
        best_file_path = os.path.join(cluster_csv_path, best_cluster_files[0])
        best_file = pd.read_csv(best_file_path)
        print_fancy_title("-------Starting Classification-------")
        print(f"Best cluster file: {best_file_path}")
        print_fancy_title("Machine Learning Models Tuning")
        results = run_pipeline(best_file, cluster_num=int(best_file['Cluster'].iloc[-1]))
    else:
        best_file_path = os.path.join(cluster_csv_path, best_cluster_files[0])
        best_file = pd.read_csv(best_file_path)
        print(f"Best cluster file: {best_file_path}")

    # Save times
    with open('times.json', 'w') as json_file:
        json.dump(times, json_file, indent=4)
    
    # Save times to CSV
    times_df = pd.DataFrame(times.items(), columns=['Step', 'Time (s)'])
    times_df.to_csv(f'./{config["output_path"]}/times.csv', index=False)


if __name__ == '__main__':
    # Reading config file from the mounted directory
    try:
        with open('Data/config.json', 'r') as f:
            config = f.read()
    except FileNotFoundError:
        with open('config.json', 'r') as f:
            config = f.read()
    try:
        config = json.loads(config)
    except json.JSONDecodeError as e:
        print(f"Error reading configuration file: {e}")
        exit(1)
    main()