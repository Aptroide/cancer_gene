# Cancer-Gene Association Analysis

## Overview

This project analyzes genetic associations between cancer and other health conditions (such as depression or Alzheimer's disease). It automatically retrieves mutation data from cBioPortal, processes frequency information, and performs clustering analysis to identify meaningful patterns.

## Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (optional, for containerized execution)
- Internet connection (for cBioPortal API access)

## Installation

### Option 1: Python Environment

1. Create a Python virtual environment:
   - Linux and macOS:
   ```bash
   python3 -m venv myenv
   ```
   - Windows:
   ```bash
   python -m venv myenv
   ```

2. Activate the virtual environment:
   - Linux and macOS:
   ```bash
   source myenv/bin/activate
   ```
   - Windows:
     - PowerShell:
     ```bash
     .\myenv\Scripts\Activate.ps1
     ```
     - Command Prompt:
     ```bash
     myenv\Scripts\activate
     ```

3. Clone the repository:
   ```bash
   git clone https://github.com/Aptroide/cancer_gene
   cd cancer_gene
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application (suppressing GPU warnings):
   - Linux and macOS:
   ```bash
   python3 main.py 2>/dev/null
   ```
   - Windows:
     - PowerShell:
     ```bash
     python main.py 2>$null
     ```
     - Command Prompt:
     ```bash
     python main.py 2>nul
     ```
   
   **Note:** If warning messages are not a concern, you may simply run `python main.py`

### Option 2: Docker Deployment

Ensure Docker and Docker Compose are installed on your system. For installation instructions, visit the [Docker Desktop Documentation](https://docs.docker.com/desktop/).

1. Download the docker-compose.yml file:
   ```bash
   curl -O https://raw.githubusercontent.com/Aptroide/cancer_gene/main/docker-compose.yml
   ```

2. Create the required directories:
   - Create a `./Data` directory to store configuration files and input data
   - Create a `./Results` directory where output files will be saved
   
   Both directories should be at the same level as the downloaded docker-compose.yml file.

3. Build and start the container:
   ```bash
   docker-compose up
   ```

## Configuration

The `./Data` directory should contain your configuration file and input data files.

Configure the analysis parameters in the `Data/config.json` file:

```json
{
    "file_path": "Data/<Condition Frequencies File.csv>",
    "studies_path": "Data/<study ids.txt>",
    "output_path": "Results",
    "num_cancer_studies": 500,
    "num_genes_per_study": null,
    "your_study_name": "Condition_Name",
    "classification": false
}
```

### Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `file_path` | Path to CSV file containing condition-related genes (requires two columns: gene names and their frequencies) |
| `studies_path` | Path to text file with cBioPortal study IDs (one per line). Use `null` to fetch all available studies |
| `output_path` | Directory where result visualizations will be saved |
| `num_cancer_studies` | Maximum number of cancer studies to analyze (use lower values like 10 for testing) |
| `num_genes_per_study` | Maximum number of genes to analyze per study (use `null` to include all genes) |
| `your_study_name` | Name of your condition study (e.g., "Alzheimer", "Depression") |
| `classification` | Set to `true` to train ML and NN models for the best cluster files, `false` to disable this feature |

### Input File Requirements

1. **Condition Gene Data (`file_path`)**:
   - Must be a CSV file with two columns
   - Example format:
     ```
     Gene,Frequency
     APP,0.45
     PSEN1,0.32
     PSEN2,0.28
     ...
     ```

2. **Study IDs (`studies_path`)**:
   - Text file with one cBioPortal study ID per line
   - Example:
     ```
     blca_tcga
     brca_tcga
     coadread_tcga
     ...
     ```

## Results

### Output Files

Analysis results are saved in the `Results/ClustersCSV` directory:

| File | Description |
|------|-------------|
| `gene_comparison.csv` | Combined and filtered frequency data |
| `K-means_labels_<method>.csv` | K-means clustering results for different dimensionality reduction methods (UMAP, KPCA, t-SNE) |
| `DBSCAN_labels_<method>.csv` | DBSCAN clustering results for different methods |
| `times.csv` | Processing time statistics |

### Output Visualizations

Clustering visualizations are saved to the `Results/Figures` directory in PNG format:

| File | Description |
|------|-------------|
| `kmeans_<method>.png` | K-means clustering visualizations |
| `dbscan_<method>.png` | DBSCAN clustering visualizations |

**Note:** `<method>` can be: `kpca`, `tsne`, or `umap`.

## Troubleshooting

1. **API Connection Issues**: 
   - Ensure you have a stable internet connection
   - Verify that the cBioPortal API is accessible (https://www.cbioportal.org/api/v2/api-docs)

2. **Missing Results**:
   - Confirm that your input files follow the required format
   - Check that the specified output directories exist and are writable

3. **Docker Issues**:
   - Verify that the Docker service is running
   - Ensure required ports are available (port 8000 is used by default)

## References

- cBioPortal: https://www.cbioportal.org/
