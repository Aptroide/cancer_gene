from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
from rich import print

def dunn_index(X, labels):
    """
    Computes the Dunn index:
      Dunn = (minimum inter-cluster distance) / (maximum intra-cluster distance).
    Higher values indicate better clustering.
    """
    unique_clusters = np.unique(labels)
    min_intercluster_dist = np.inf
    max_intracluster_dist = 0

    for i in unique_clusters:
        points_i = X[labels == i]
        if len(points_i) > 1:
            intra_dists = cdist(points_i, points_i)
            max_intracluster_dist = max(max_intracluster_dist, np.max(intra_dists))
        for j in unique_clusters:
            if i != j:
                points_j = X[labels == j]
                inter_dists = cdist(points_i, points_j)
                min_intercluster_dist = min(min_intercluster_dist, np.min(inter_dists))

    if max_intracluster_dist == 0:
        return np.inf
    return min_intercluster_dist / max_intracluster_dist

def xie_beni_index(X, labels):
    """
    Computes the Xie–Beni index:
      XB = (sum of squared distances of points from their cluster centroids) /
           (n_samples * (minimum squared distance between any two centroids)).
    Lower values indicate more compact and well-separated clusters.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = X.shape[0]
    
    centroids = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
    
    sse = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = cluster_points.mean(axis=0)
        sse += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
    
    if n_clusters < 2:
        return np.inf
    centroid_dists = cdist(centroids, centroids, metric='euclidean')
    np.fill_diagonal(centroid_dists, np.inf)
    min_centroid_dist = np.min(centroid_dists)
    
    if min_centroid_dist == 0:
        return np.inf
    
    return sse / (n_samples * (min_centroid_dist ** 2))

def c_kmeans(data, k_min=2, k_max=10):
    """
    Determines the optimal number of clusters (k) for KMeans by computing several metrics
    and then selecting the candidate that is chosen by the majority of metrics.
    
    Metrics used:
      - Silhouette Score (higher is better)
      - Calinski–Harabasz Score (higher is better)
      - Davies–Bouldin Score (lower is better)
      - Dunn Index (higher is better)
      - Xie–Beni Index (lower is better)
    
    Parameters:
      data   : Data to cluster.
      method : A string parameter (passed from main; not used in the computation here).
      k_min  : Minimum k to test (default=2).
      k_max  : Maximum k to test (default=10).
    
    Returns:
      optimal_k : The chosen number of clusters based on majority vote.
    """
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    dunn_scores = []
    xie_beni_scores = []
    
    # Compute metrics for each candidate k
    for k in range(k_min, k_max+1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(data)
        
        silhouette = silhouette_score(data, labels)
        calinski = calinski_harabasz_score(data, labels)
        davies = davies_bouldin_score(data, labels)
        dunn = dunn_index(data, labels)
        xie_beni = xie_beni_index(data, labels)
        
        silhouette_scores.append(silhouette)
        calinski_scores.append(calinski)
        davies_bouldin_scores.append(davies)
        dunn_scores.append(dunn)
        xie_beni_scores.append(xie_beni)
    
    # For metrics where higher is better, choose candidate with maximum value.
    best_silhouette = np.argmax(silhouette_scores) + k_min
    best_calinski   = np.argmax(calinski_scores) + k_min
    best_dunn       = np.argmax(dunn_scores) + k_min
    # For metrics where lower is better, choose candidate with minimum value.
    best_davies     = np.argmin(davies_bouldin_scores) + k_min
    best_xie_beni   = np.argmin(xie_beni_scores) + k_min
    
    metric_choices = [best_silhouette, best_calinski, best_dunn, best_davies, best_xie_beni]
    
    # Count how many metrics chose each candidate k
    counts = Counter(metric_choices)
    
    # If any candidate gets at least 3 votes, return it.
    for candidate, count in counts.items():
        if count >= 3:
            print(f"Majority decision: {candidate} (with {count} votes)")
            return candidate
    
    # If no candidate has a majority, return the candidate with the most votes.
    best_candidate, best_count = counts.most_common(1)[0]
    print(f"No majority; defaulting to candidate: {best_candidate} (with {best_count} votes)")
    
    return best_candidate

def plot_kmeans_silhouette_plotly(proj_2d, df_transpuesto, range_n_clusters, image_name, output_path):
    """
    Create silhouette analysis plots for KMeans clustering visualization using Plotly
    and add a 'Grupo' column to the original DataFrame based on the best clustering.

    Parameters:
    proj_2d : array-like of shape (n_samples, 2)
        The 2D projected data to cluster
    df_transpuesto : pandas DataFrame
        The original dataframe containing the Cancer labels
    range_n_clusters : array-like
        The range of number of clusters to try

    Returns:
    dict: Dictionary with cluster labels and silhouette scores for each n_clusters value
    pandas.DataFrame: Original DataFrame with an added 'Grupo' column for the best clustering
    """
    results = {}
    summary = []  # Lista para almacenar los resultados

    for n_clusters in range_n_clusters:
        # Inicializar el clusterer
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(proj_2d)

        # Calcular las puntuaciones de silueta
        silhouette_avg = silhouette_score(proj_2d, cluster_labels)
        sample_silhouette_values = silhouette_samples(proj_2d, cluster_labels)

        # Almacenar resultados
        results[n_clusters] = {
            'labels': cluster_labels,
            'silhouette_score': silhouette_avg
        }

        # Añadir al resumen
        summary.append({
            'n_clusters': n_clusters,
            'silhouette_avg': silhouette_avg
        })

        # Crear subplots
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Silhouette Plot", "Clustered Data"),
                            column_widths=[0.4, 0.6])

        # Silhouette plot
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            # Añadir área rellenada para cada cluster
            y_pts = np.arange(y_lower, y_upper)
            fig.add_trace(
                go.Scatter(
                    x=ith_cluster_silhouette_values,
                    y=y_pts,
                    fill='tozerox',
                    name=f'Cluster {i}',
                    showlegend=False
                ),
                row=1, col=1
            )

            y_lower = y_upper + 10

        # Añadir línea vertical para la puntuación de silueta promedio
        fig.add_vline(x=silhouette_avg, line_dash="dash", line_color="red",
                     annotation_text=f"Average silhouette score: {silhouette_avg:.3f}",
                     row=1, col=1)

        # Scatter plot con clusters
        centers = clusterer.cluster_centers_

        # Crear scatter plot coloreado por clusters
        fig.add_trace(
            go.Scatter(
                x=proj_2d[:, 0],
                y=proj_2d[:, 1],
                mode='markers',
                marker=dict(
                    color=cluster_labels,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Cluster')
                ),
                text=[f'Cluster {label}<br>Cancer: {cancer}'
                      for label, cancer in zip(cluster_labels, df_transpuesto.Cancer)],
                hoverinfo='text',
                name='Data points'
            ),
            row=1, col=2
        )

        # Añadir centros de clusters
        fig.add_trace(
            go.Scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode='markers+text',
                marker=dict(
                    color='white',
                    size=15,
                    line=dict(color='black', width=2),
                    symbol='diamond'
                ),
                text=[f'Center {i}' for i in range(n_clusters)],
                name='Centroids',
                showlegend=True
            ),
            row=1, col=2
        )

        # Actualizar layout
        fig.update_layout(
            title_text=f"Silhouette Analysis for KMeans Clustering (n_clusters = {n_clusters})",
            height=600,
            width=1200,
            showlegend=True
        )

        # Actualizar ejes
        fig.update_xaxes(title_text="Silhouette coefficient values", range=[-0.1, 1], row=1, col=1)
        fig.update_yaxes(showticklabels=False, title_text="Cluster label", row=1, col=1)
        fig.update_xaxes(title_text="First dimension", row=1, col=2)
        fig.update_yaxes(title_text="Second dimension", row=1, col=2)

        current_dir = os.getcwd()
        image_path = os.path.join(current_dir, f"{output_path}/Figures", f"Kmeans_{image_name}.png")
        fig.write_image(image_path)

    # Convertir el resumen a DataFrame
    results_df = pd.DataFrame(summary)

    # Identificar el n_clusters con la mejor puntuación de silueta
    best_n_clusters = results_df.loc[results_df['silhouette_avg'].idxmax(), 'n_clusters']
    best_silhouette = results_df['silhouette_avg'].max()
    print(f"Silhouette Score: {best_silhouette:.3f}")

    # Obtener las etiquetas del mejor clustering
    best_labels = results[best_n_clusters]['labels']
    # Crear una copia del DataFrame original para no modificarlo en su lugar
    df_with_clusters = df_transpuesto.copy()

    # Agregar la columna 'Grupo' con las etiquetas del mejor clustering
    df_with_clusters['Cluster'] = best_labels


    # Reordenar las columnas para que 'Cluster' esté justo después de 'Cancer'
    cols = list(df_with_clusters.columns)
    cancer_idx = cols.index('Cancer')
    # Insertar 'Cluster' después de 'Cancer'
    cols.insert(cancer_idx + 1, cols.pop(cols.index('Cluster')))
    df_with_clusters = df_with_clusters[cols]


    return results, results_df, df_with_clusters, best_silhouette