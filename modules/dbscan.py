import itertools
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import os
from rich import print

def get_scores_and_labels(combinations, X):
    scores =[]
    all_labels_list = []

    for i, (eps, num_samples) in enumerate(combinations):
        dbscan = DBSCAN(eps=eps, min_samples=num_samples)
        labels = dbscan.fit_predict(X)
        labels_set = set(labels)
        num_clusters = len(labels_set) - (1 if -1 in labels_set else 0)
        if num_clusters < 2 or num_clusters > 50:
            scores.append(-10)
            all_labels_list.append('bad')
            c = (eps, num_samples)
            # Skip combinations with too few or too many clusters
            continue
        scores.append(silhouette_score(X, labels))
        all_labels_list.append(labels)

    best_index = np.argmax(scores)
    best_parameters = combinations[best_index]
    best_labels = all_labels_list[best_index]
    best_score = scores[best_index]

    return {'best_epsilons': best_parameters[0],
                    'best_min_samples': best_parameters[1],
                    'best_score': best_score}


def optimize_dbscan(proj2d, df_transpuesto, method, output_path):
        # Define values for eps and min_samples
        eps_values = np.linspace(0.1, 1.3, num=100)
        min_samples_values = np.arange(2, 20, step=3)

        # Generate all possible combinations of (eps, min_samples)
        combinations = list(itertools.product(eps_values, min_samples_values))

        # Get the best values of epsilon and min_samples using get_scores_and_labels
        best_dict = get_scores_and_labels(combinations, proj2d)

        # Extract the best values of epsilon and min_samples
        eps_values = [best_dict['best_epsilons']]
        min_samples_values = [best_dict['best_min_samples']]

        # Get the results and DBSCAN labels
        results, dbscan_labels, sScore = plot_dbscan_silhouette_plotly(proj2d, df_transpuesto, eps_values, min_samples_values, method, output_path)

        return results, dbscan_labels, sScore


def plot_dbscan_silhouette_plotly(proj_2d, df_transpuesto, eps_values, min_samples_values, image_name, output_dir):
        """
        Create silhouette analysis plots for DBSCAN clustering visualization using Plotly.
        This function returns the DataFrame corresponding to the best silhouette score.
        """

        results = {}
        best_silhouette_score = -1  # Start with a very low value
        best_eps = None
        best_min_samples = None
        best_cluster_labels = None
        best_sample_silhouette_values = None

        # Create a copy of the DataFrame to avoid modifying the original
        df_augmented = df_transpuesto.copy()

        # First, find the best parameters without generating visualizations
        for eps in eps_values:
                for min_samples in min_samples_values:
                        # Initialize the DBSCAN clusterer
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        cluster_labels = dbscan.fit_predict(proj_2d)

                        # Calculate silhouette scores
                        if len(set(cluster_labels)) > 1:  
                                silhouette_avg = silhouette_score(proj_2d, cluster_labels)
                                sample_silhouette_values = silhouette_samples(proj_2d, cluster_labels)
                        else:
                                silhouette_avg = -1
                                sample_silhouette_values = np.zeros(len(proj_2d))

                        # Store the results
                        results[(eps, min_samples)] = {
                                'labels': cluster_labels,
                                'silhouette_score': silhouette_avg
                        }

                        # Check if the silhouette score is the best
                        if silhouette_avg > best_silhouette_score:
                                best_silhouette_score = silhouette_avg
                                best_eps = eps
                                best_min_samples = min_samples
                                best_cluster_labels = cluster_labels
                                best_sample_silhouette_values = sample_silhouette_values

                                # Update the DataFrame with cluster labels
                                df_augmented['Cluster'] = cluster_labels

                                # Reorder columns so that 'Cluster' is right after 'Cancer'
                                cols = list(df_augmented.columns)
                                cancer_idx = cols.index('Cancer')
                                # Insert 'Cluster' after 'Cancer'
                                cols.insert(cancer_idx + 1, cols.pop(cols.index('Cluster')))
                                df_augmented = df_augmented[cols]

        # Now generate visualization only for the best parameters
        if best_eps is not None and best_min_samples is not None:
                # Create subplots
                fig = make_subplots(rows=1, cols=2,
                                    subplot_titles=("Silhouette Plot", "Clustered Data"),
                                    column_widths=[0.4, 0.6])

                # Silhouette plot
                y_lower = 10
                unique_clusters = np.unique(best_cluster_labels)
                for i in unique_clusters:
                        if i == -1:
                                continue  # Skip noise points
                        ith_cluster_silhouette_values = best_sample_silhouette_values[best_cluster_labels == i]
                        ith_cluster_silhouette_values.sort()

                        size_cluster_i = ith_cluster_silhouette_values.shape[0]
                        y_upper = y_lower + size_cluster_i

                        # Add filled area for silhouette values of each cluster
                        y_pts = np.arange(y_lower, y_upper)
                        fig.add_trace(
                                go.Scatter(
                                        x=ith_cluster_silhouette_values,
                                        y=y_pts,
                                        fill='tozerox',
                                        mode='none',
                                        name=f'Cluster {i}',
                                        showlegend=False
                                ),
                                row=1, col=1
                        )

                        y_lower = y_upper + 10

                # Add vertical line for the average silhouette score
                fig.add_vline(x=best_silhouette_score, line_dash="dash", line_color="red",
                            annotation_text=f"Average silhouette score: {best_silhouette_score:.3f}",
                            row=1, col=1)

                # Scatter plot with clusters
                scatter_df = pd.DataFrame({
                        'Dim1': proj_2d[:, 0],
                        'Dim2': proj_2d[:, 1],
                        'Cancer Type': df_transpuesto['Cancer'],
                        'Cluster': best_cluster_labels.astype(str)  # Convert to string for color handling
                })

                # Define colors for clusters, including noise
                unique_clusters_str = scatter_df['Cluster'].unique()
                color_discrete_map = {str(cluster): px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                                                            for i, cluster in enumerate(unique_clusters_str)}

                fig_scatter = px.scatter(
                        scatter_df,
                        x='Dim1', y='Dim2',
                        color='Cluster',
                        color_discrete_map=color_discrete_map,
                        hover_data={'Cancer Type': True, 'Cluster': True},
                        labels={'Cluster': 'Cluster'}
                )

                # Add scatter plot traces to the subplot
                for trace in fig_scatter.data:
                        fig.add_trace(trace, row=1, col=2)

                # Update layout
                fig.update_layout(
                        title_text=f"Silhouette Analysis for DBSCAN Clustering (eps = {best_eps}, min_samples = {best_min_samples})",
                        height=600,
                        width=1200
                )

                # Update axes
                fig.update_xaxes(title_text="Silhouette coefficient values", range=[-0.1, 1], row=1, col=1)
                fig.update_yaxes(showticklabels=False, title_text="Cluster label", row=1, col=1)
                fig.update_xaxes(title_text="First dimension", row=1, col=2)
                fig.update_yaxes(title_text="Second dimension", row=1, col=2)

                current_dir = os.getcwd()
                fig.write_image(os.path.join(current_dir, f"{output_dir}/Figures", f"dbscan_{image_name}.png"))
                # fig.write_html(os.path.join(current_dir, "Results/Figures", f"dbscan_{image_name}_method.html"))

        # After the loop, df_augmented now contains cluster labels corresponding to the best average silhouette score
        print(f"Silhouette Score: {best_silhouette_score:.3f}")

        # Return the results and the DataFrame corresponding to the best silhouette score
        return results, df_augmented, best_silhouette_score
