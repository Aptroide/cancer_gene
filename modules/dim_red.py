from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from umap import UMAP
import warnings
from rich import print

def dim_reduction(df, features):
  # Suppress the specific warning
  warnings.filterwarnings('ignore', category=UserWarning, module='umap')
  # Aplicar UMAP
  umap_2d = UMAP(n_components=2, init='random', random_state=42)
  proj_2d = umap_2d.fit_transform(features)

  # Aplicar KernelPCA para reducir la dimensionalidad a 2 componentes
  kpca = KernelPCA(n_components=2, kernel='rbf')
  kpca_features = kpca.fit_transform(features)

  # Aplicar TSNE para reducir la dimensionalidad a 2 componentes
  tsne = TSNE(n_components=2, random_state=42)
  tsne_features = tsne.fit_transform(features)

  return proj_2d, kpca_features, tsne_features
