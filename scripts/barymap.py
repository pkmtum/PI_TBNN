"""
Postprocessing file to create barycentric map from RANS data. Class instant is created
and contains all data and plotting methods
"""

import numpy as np
import torch as th
import scipy as sp
from scipy.interpolate import interp1d
import seaborn as sns
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from scripts.utilities import mask_boundary_points


def anisotropy(rs, k):
    """
    calculate normalized anisotropy tensor
    :param rs: reynolds stress tensor (Nx3x3)
    :param k: tke (N)
    :return:
    """
    k = th.maximum(k, th.tensor(1e-8))
    b = rs[:] / (2 * k.unsqueeze(1).unsqueeze(2)) - 1 / 3 * th.eye(3).unsqueeze(0).expand(k.shape[0], 3, 3)
    return b


def plot_barycentric_triangle(axis, x_lim=np.array([1, 0, .5]), y_lim=np.array([0, 0, np.sin(np.pi / 3)])):
    """plot the barycentric triangle to an axis"""
    axis.plot(np.array([*x_lim, x_lim[0]]), np.array([*y_lim, y_lim[0]]), color='black', linewidth=1.5)


def plot_periodic_hills_boundaries(axis, linewidth=1, color='black'):
    """plot boundaries of periodic hill case"""
    x = (1.929 / 54) * np.array([0., 0.1, 9., 14., 20., 30., 40., 54.9, 54.])
    y = (1.929 / 54) * np.array([28., 28., 27., 24., 19., 11., 4., 0., 0.])
    x_new = np.linspace(0, 1.929, 100)
    f = interp1d(x, y, kind='cubic')
    axis.plot(x_new, f(x_new), linewidth=linewidth, color=color)
    axis.plot(9 - x_new, f(x_new), linewidth=linewidth, color=color)
    axis.plot([1.929, 9 - 1.929], [0, 0], linewidth=linewidth, color=color)
    axis.plot([0, 9], [3.035, 3.035], linewidth=linewidth, color=color)


def plot_conv_div_channel_7900_boundaries(axis, path, linewidth=1, color='black'):
    """plot boundaries of converging diverging channel case"""
    x_new = np.linspace(0, 12.5664, 200)
    surface_points = np.genfromtxt(path, delimiter=' ', skip_header=3,
                                   skip_footer=0)  # names=["X", "Y"])
    f = interp1d(surface_points[:, 0], surface_points[:, 1], kind='cubic', fill_value='extrapolate')
    axis.plot(x_new, f(x_new), linewidth=linewidth, color=color)
    axis.plot([0, 12.5664], [2., 2.], linewidth=linewidth, color=color)


class BarMap:

    def __init__(self):  # (self, path)
        self.RS = np.array([])  # np.array(tn.load(path + '/RS-torch.th')).reshape((-1, 3, 3))
        self.k = np.array([])  # np.array(tn.load(path + '/k-torch.th'))
        self.cell_centers = np.array([])  # np.array(tn.load(path + '/cellCenters-torch.th'))
        self.eig_val_sorted = np.array([])
        self.eig_vec_sorted = np.array([])
        self.b = np.array([])
        self.isb = False
        self.c = np.array([])
        self.x_bar = np.array([])
        self.y_bar = np.array([])
        self.x_lim = np.array([1, 0, .5])
        self.y_lim = np.array([0, 0, np.sin(np.pi / 3)])
        # self.calculate_barycentric_coordinates()

    def load_from_path(self, path):
        """Load the Data From an OpenFoam flow case.
        path should point to the time directory of interest"""
        self.RS = np.array(th.load(path + '/RS-torch.th')).reshape((-1, 3, 3))
        self.k = np.array(th.load(path + '/k-torch.th'))
        self.cell_centers = np.array(th.load(path + '/cellCenters-torch.th'))
        if self.cell_centers.shape[1] == 3:  # in case cell_centers still contains all 3 coords
            self.cell_centers = np.array([self.cell_centers[:, 0], self.cell_centers[:, 1]]).T

    def load_from_variable(self, b, cell_centers):
        """load in b tensor and directly and skip its calculation
        also load cell centers"""
        self.b = b
        self.isb = True
        self.cell_centers = cell_centers
        if self.cell_centers.shape[1] == 3:  # in case cell_centers still contains all 3 coords
            self.cell_centers = np.array([self.cell_centers[:, 0], self.cell_centers[:, 1]]).T

    def calculate_barycentric_coordinates(self):
        """calculate the barycentric coordinates for the given dataset"""

        if not self.isb:
            # filter out data points where RS-tensor is not diagonalizable
            mask = []
            count = 0

            for i in range(self.RS.shape[0]):
                if np.sum(self.RS[i].diagonal() ** 2) == 0.0:
                    mask.append(False)
                    count += 1
                else:
                    mask.append(True)
            print('removing %d data points ...' % count)

            # removing the data points
            self.RS = self.RS[mask]
            self.k = self.k[mask]
            self.cell_centers = self.cell_centers[mask]
            print('successfully removed')

            # computing b
            b = anisotropy(th.tensor(self.RS), th.tensor(self.k))
            print(b.shape)

        else:
            b = self.b

        # spectral decomposition of b
        eig_val, eig_vec = np.linalg.eig(b)

        # sorting eigenvalues
        idx = np.argsort((-eig_val))
        self.eig_val_sorted = np.zeros(eig_val.shape)
        self.eig_vec_sorted = np.zeros(eig_vec.shape)
        for i, val in enumerate(idx):
            self.eig_val_sorted[i] = eig_val[i, val]
            self.eig_vec_sorted[i] = eig_vec[i, :, val]

        # computing barycentric coefficients
        self.c = np.array([(self.eig_val_sorted[:, 0] - self.eig_val_sorted[:, 1]),
                           2 * (self.eig_val_sorted[:, 1] - self.eig_val_sorted[:, 2]),
                           3 * self.eig_val_sorted[:, 2] + 1]).T

        # coordinates of a in spectral tensor basis
        self.x_bar = self.c.dot(self.x_lim)
        self.y_bar = self.c.dot(self.y_lim)

    def plot_data_points(self, axis, color=sns.color_palette()[0], **kwargs):
        """plot all data points in barycentric triangle"""
        axis.scatter(self.x_bar, self.y_bar, color=color, **kwargs)
        return axis

    def plot_data_line(self, axis, color=sns.color_palette()[0], **kwargs):
        """plot all data points in barycentric triangle and connects them with line"""
        axis.plot(self.x_bar, self.y_bar, 'o-', color=color, **kwargs)
        return axis

    @staticmethod
    def plot_triangle(axis):
        """plot the boundary of realizable turbulence states"""
        plot_barycentric_triangle(axis)

    def get_colormap(self, axis, normalized=True):
        """plot a colormap in a given axis"""
        # compute coefficients from coordinates
        y, x = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        c_3 = 2 / np.sqrt(3) * y
        c_1 = x - .5 * c_3
        c_2 = 1 - c_3 - c_1

        if normalized:
            norm_c = np.array([c_1, c_2, c_3]).max(axis=0)
            c_1 = c_1 / norm_c
            c_2 = c_2 / norm_c
            c_3 = c_3 / norm_c

        # creating image
        c_grid = np.array([c_1, c_2, c_3]).T
        image = axis.imshow(c_grid, interpolation='gaussian', origin='lower', extent=([0, 1, 0, 1]))

        # cutting out triangle
        triangle = Polygon(np.array([self.x_lim, self.y_lim]).T, transform=axis.transData)
        image.set_clip_path(triangle)

    def plot_on_geometry(self, axis, flowcase='PH', extent=None, resolution=0.005, cut_boundary=False, wall_color=0.85):
        """plot barycentric colormap on geometry of flow car.
        inputs:
            axis - matplotlib pyplot handle where to plot
            boundaries - set rectangle that covers fluid domain [x_min, x_max, y_min, y_min]
            resolution - dx between data points of interpolated grid
        """
        # interpolating data points to grid
        if extent is None:
            # extent = [0., 1., 0., 1.]
            extent = [self.cell_centers[:, 0].min(), self.cell_centers[:, 0].max(),
                      self.cell_centers[:, 1].min() , self.cell_centers[:, 1].max()]
        x_grid, y_grid = np.meshgrid(np.arange(*extent[0:2], resolution), np.arange(*extent[2:4], resolution))
        c_int = np.zeros((len(x_grid.flatten()), 3))
        for i in range(3):
            c_int[:, i] = sp.interpolate.griddata(self.cell_centers,
                                                  self.c[:, i],
                                                  (x_grid.flatten(), y_grid.flatten()),
                                                  method='linear') #, fill_value='nan')

        c_int = c_int/np.expand_dims(c_int.max(axis=1), axis=1)
        c_int = c_int.reshape(x_grid.shape[0], x_grid.shape[1], 3)

        if cut_boundary:
            mask = mask_boundary_points(x_grid, y_grid, flowcase)
            c_int[~mask] = wall_color

        # plotting
        axis.imshow(c_int, extent=extent, origin='lower', interpolation='gaussian')
        # plot_periodic_hills_boundaries(axis, color='white')


if __name__ == '__main__':

    # example on how to use the barycentric map
    path = '/home/leonriccius/Documents/Fluid_Data/tensordata/SquareDuct/2000'

    # load in anisotropy tensor b
    b_dns = th.load(os.sep.join([path, 'b_dns-torch.th']))
    b_rans = th.load(os.sep.join([path, 'b_rans-torch.th']))
    grid = th.load(os.sep.join([path, 'grid-torch.th']))

    # create BarMap objects and compute barycentric coordinates
    barm_rans = BarMap()
    barm_rans.load_from_variable(b_rans, grid[:, 1:3])
    barm_rans.calculate_barycentric_coordinates()

    # plot and show
    fig, ax = plt.subplots()
    barm_rans.plot_on_geometry(ax)
    fig.show()
