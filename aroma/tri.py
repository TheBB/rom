from nutils import cache, plot
import numpy as np
from scipy.spatial import cKDTree


def compute_renumber(points, edges, mergetol):
    npoints = len(points)
    onedge = np.zeros(npoints, dtype=bool)
    onedge[edges] = True
    index, = onedge.nonzero()
    for i, j in sorted(cKDTree(points[onedge]).query_pairs(mergetol)):
        assert i < j
        index[j] = index[i]
    renumber = np.arange(npoints)
    renumber[onedge] = index
    return renumber

def triangulate(points, mergetol=0):
    triangulate_bezier = cache.Wrapper(plot._triangulate_bezier)
    npoints = 0
    triangulation = []
    edges = []
    for epoints in points:
        assert epoints.ndim == 2
        assert epoints.shape[-1] == 2
        etri, ehull = triangulate_bezier(len(epoints))
        triangulation.append(npoints + etri)
        edges.append(npoints + ehull)
        npoints += len(epoints)
    triangulation = np.concatenate(triangulation, axis=0)
    edges = np.concatenate(edges, axis=0)
    points = np.concatenate(points, axis=0)

    if mergetol:
        renumber = compute_renumber(points, edges, mergetol)
        triangulation = renumber[triangulation]
        edges = np.sort(renumber[edges], axis=1)
        edges = edges[np.lexsort(edges.T)]
        edges = edges[np.concatenate([[True], np.diff(edges, axis=0).any(axis=1)])]

    return triangulation, edges
