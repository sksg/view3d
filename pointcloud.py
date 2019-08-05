import numpy as np
import scipy.spatial
import plyfile
import warnings
import os


class pointcloud(object):
    """pointcloud encapsulates positions, normals, and colors.

    The class can read and write Standford .ply files"""

    def __init__(self, positions=None, colors=None, normals=None):
        super(pointcloud, self).__init__()
        self.positions = positions
        self.colors = colors
        self.normals = normals

    def writePLY(self, filename, ascii=False):
        dtype = []
        N = -1
        if self.positions is not None:
            N = len(self.positions)
            dtype += [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        if self.colors is not None:
            N = len(self.colors) if N == -1 else N
            dtype += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        if self.normals is not None:
            N = len(self.normals) if N == -1 else N
            dtype += [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]

        error_msg = "Lengths of positions, colors, and normals must match."
        if self.positions is not None and N != len(self.positions):
            raise RuntimeError(error_msg)
        if self.colors is not None and N != len(self.colors):
            raise RuntimeError(error_msg)
        if self.normals is not None and N != len(self.normals):
            raise RuntimeError(error_msg)

        vertex = np.zeros((N,), dtype=dtype)

        if self.positions is not None:
            vertex['x'] = self.positions[:, 0].astype('f4')
            vertex['y'] = self.positions[:, 1].astype('f4')
            vertex['z'] = self.positions[:, 2].astype('f4')

        if self.colors is not None:
            # assuming RGB format
            vertex['red'] = self.colors[:, 0].astype('u1')
            vertex['green'] = self.colors[:, 1].astype('u1')
            vertex['blue'] = self.colors[:, 2].astype('u1')

        if self.normals is not None:
            vertex['nx'] = self.normals[:, 0].astype('f4')
            vertex['ny'] = self.normals[:, 1].astype('f4')
            vertex['nz'] = self.normals[:, 2].astype('f4')

        vertex = plyfile.PlyElement.describe(vertex, 'vertex')

        ext = filename.split('.')[-1]
        if ext != "ply" and ext != "PLY":
            filename = filename + '.ply'
        plyfile.PlyData([vertex], text=ascii).write(filename)
        return self

    def readPLY(self, filename):
        self.__init__()

        vertex = plyfile.PlyData.read(filename)['vertex']

        with warnings.catch_warnings():
            # numpy does not like to .view() into structured array
            warnings.simplefilter("ignore")

            if all([p in vertex.data.dtype.names for p in ('x', 'y', 'z')]):
                position_data = vertex.data[['x', 'y', 'z']]
                N = len(position_data.dtype.names)
                self.positions = position_data.view((position_data.dtype[0],
                                                     N))

            colored = all([p in vertex.data.dtype.names
                           for p in ('red', 'green', 'blue')])
            if colored:
                color_data = vertex.data[['red', 'green', 'blue']]
                N = len(color_data.dtype.names)
                self.colors = color_data.view((color_data.dtype[0], N))

            if all([p in vertex.data.dtype.names for p in ('nx', 'ny', 'nz')]):
                normal_data = vertex.data[['nx', 'ny', 'nz']]
                N = len(normal_data.dtype.names)
                self.normals = normal_data.view((normal_data.dtype[0], N))
        return self
