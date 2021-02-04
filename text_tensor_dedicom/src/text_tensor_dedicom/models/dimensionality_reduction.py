import umap


def umap_transform(A):
    reducer = umap.UMAP()
    A_fit = reducer.fit_transform(A)
    return A_fit

