from matplotlib import tri
from nutils import cache, plot
import numpy as np
from scipy.spatial import cKDTree


class Triangulation(tri.Triangulation):

    @classmethod
    def from_tri(cls, tri):
        return cls(tri.x, tri.y, tri.triangles, tri.mask)

    def __add__(self, other):
        if isinstance(other, Triangulation):
            return Triangulation(self.x + other.x, self.y + other.y, self.triangles, self.mask)
        return Triangulation(self.x + other, self.y + other, self.triangles, self.mask)

    def __mul__(self, other):
        return Triangulation(self.x * other, self.y * other, self.triangles, self.mask)


class Triangulator:

    def __init__(self, mergetol=0):
        self.mergetol = mergetol

    def compute_renumber(self, points, edges):
        npoints = len(points)
        onedge = np.zeros(npoints, dtype=bool)
        onedge[edges] = True
        index, = onedge.nonzero()
        for i, j in sorted(cKDTree(points[onedge]).query_pairs(self.mergetol)):
            assert i < j
            index[j] = index[i]
        self.renumber = np.arange(npoints)
        self.renumber[onedge] = index

    def triangulate(self, points):
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

        if self.mergetol and not hasattr(self, 'renumber'):
            self.compute_renumber(points, edges)
        if self.mergetol:
            triangulation = self.renumber[triangulation]
            edges = np.sort(self.renumber[edges], axis=1)
            edges = edges[np.lexsort(edges.T)]
            edges = edges[np.concatenate([[True], np.diff(edges, axis=0).any(axis=1)])]

        return Triangulation.from_tri(tri.Triangulation(points[:,0], points[:,1], triangulation)), points[edges]
