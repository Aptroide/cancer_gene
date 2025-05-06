# Cancer-Gene Association Analysis

## Overview

This project analyzes genetic associations between cancer and other conditions (like depression or Alzheimer's). It automatically fetches mutation data from cBioPortal, processes frequency information, and performs clustering analysis to identify patterns.

## Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (optional, for containerized execution)
- Internet connection (for cBioPortal API access)

## Installation

### Option 1: Standard Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Aptroide/cancer_gene
   cd cancer_gene
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Docker Installation

1. Make sure Docker and Docker Compose are installed on your system.
2. Clone the repository and navigate to the project directory.
3. Build and start the container:
   ```bash
   docker-compose up --build
   ```

## Configuration

When using Docker, you must map your local directories to the container directories in the `docker-compose` file:

```yml
volumes:
  - ./Results/Figures:/usr/src/app/Results/Figures 
  - ./Results/ClustersCSV:/usr/src/app/Results/ClustersCSV 
  - ./Results:/usr/src/app/Results
  - ./Data:/usr/src/app/Data
```
Replace the paths before the colon with your actual local directory paths if different from the example above.

`./Data` folder is where you should have the configuration file and the files you wanna work with.


Configure the analysis parameters in the `Data/config.json` file:

```json
{
    "file_path": "/usr/src/app/Data/<Condition Frequencies File.csv>",
    "studies_path": "/usr/src/app/Data/<study ids.txt>",
    "output_path": "/usr/src/app/Results/Figures",
    "num_cancer_studies": 500,
    "num_genes_per_study": null,
    "your_study_name": "Condition_Name"
}
```

*Note*: The paths in this file should always use the container paths (starting with `/usr/src/app/`) when using Docker.

### Configuration Parameters:

| Parameter | Description |
|-----------|-------------|
| `file_path` | Path to CSV file containing the condition-related genes (must have a "Gene" and "Frecuencia (%)" column) |
| `studies_path` | Path to text file with cBioPortal study IDs (one per line). Use `null` to fetch all available studies. |
| `output_path` | Directory where result figures will be saved |
| `num_cancer_studies` | Maximum number of cancer studies to analyze (use lower values like 10 for testing) |
| `num_genes_per_study` | Maximum number of genes to analyze per study. Use `null` to include all genes. |
| `your_study_name` | Name of your condition study (e.g., "Alzheimer", "Depression") |

### Input File Requirements

1. **Your Condition Gene Data (`file_path`)**:
   - Must be a CSV file with at least a "Gene" column
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

## Running the Analysis



### Using Python
Change the path of `config.json` in `main.py` file.

Run the main script:
```bash
python main.py
```

### Using Docker

```bash
docker-compose up
```

## Output Files

Results are saved in the `Results` directory:

| File | Description |
|------|-------------|
| `gene_comparison.csv` | Combined and filtered frequency data |
| `kMeans_labels_*.csv` | K-means clustering results for different methods (umap, kpca, tsne) |
| `DBSCAN_labels_*.csv` | DBSCAN clustering results for different methods |
| `times.csv` | Processing time statistics |

## Visualizations

Clustering visualizations are saved to the `Results/Figures` directory in HTML format:

- `kmeans_cluster_*_method.html`: Interactive K-means clustering plots
- `dbscan_cluster_*_method.html`: Interactive DBSCAN clustering plots

## Troubleshooting

1. **API Connection Issues**: 
   - Ensure you have a stable internet connection
   - Check if cBioPortal API is accessible (https://www.cbioportal.org/api/v2/api-docs)

2. **Missing Results**:
   - Verify your input files meet the required format
   - Check the console output for error messages
   - Try with a smaller number of studies (e.g., set `num_cancer_studies` to 10)

3. **Docker Issues**:
   - Ensure Docker service is running
   - Check if ports are available (8000 is used by default)

## License

[Specify license information here]

## References

- cBioPortal: https://www.cbioportal.org/
