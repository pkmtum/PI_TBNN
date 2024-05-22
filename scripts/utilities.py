# Python script with some utilities to read an write data as well as calculating tensors
import torch as th
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy import ndimage
import datetime
import os, sys, io, contextlib
import scripts.preProcess as pre

# Default tensor type
dtype = th.DoubleTensor


def time():
    return datetime.datetime.now().strftime("%y-%m-%d_%H-%M")


class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


def bisection(f,a,b,N):
    """
    Approximate solution of f(x)=0 on interval [a,b] by bisection method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x)=0.
    a,b : numbers
        The interval in which to search for a solution. The function returns
        None if f(a)*f(b) >= 0 since a solution is not guaranteed.
    N : (positive) integer
        The number of iterations to implement.

    Returns
    -------
    x_N : number
        The midpoint of the Nth interval computed by the bisection method. The
        initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some
        midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iteration, the bisection method fails and return None.
    """

    if f(a)*f(b) >= 0:
        print("Bisection method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1,N+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)
        if f(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n)*f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Found exact solution.")
            return m_n
        else:
            print("Bisection method fails.")
            return None
    return (a_n + b_n)/2


def sigmoid_scaling(tensor):
    return (1.0 - th.exp(-tensor)) / (1.0 + th.exp(-tensor))


def mean_std_scaling(tensor, cap=2., mu=None, std=None):
    if (mu is None) & (std is None):
        rescale = True

        # calculate mean and standard deviation
        mu = th.mean(tensor, 0)
        std = th.std(tensor, 0)
    else:
        rescale = False

    # normalize tensor
    tensor = (tensor - mu) / std

    # remove outliers
    tensor[tensor > cap] = cap
    tensor[tensor < -cap] = -cap

    if rescale:
        # rescale tensors and recalculate mu and std from capped tensor
        tensor = tensor * std + mu
        mu = th.mean(tensor, 0)
        std = th.std(tensor, 0)

        # renormalize tensor after capping
        tensor = (tensor - mu) / std

    return tensor, mu, std


def cap_tensor(tensor, cap):
    tensor[tensor > cap] = cap
    tensor[tensor < -cap] = -cap
    return tensor


def get_invariants(s, r):
    """function for computation of tensor basis
        Inputs:
            s: N x 3 x 3 (N is number of data points)
            r: N x 3 x 3
        Outputs:
            invar : N x 5 (5 is number of scalar invariants)
    """

    nCells = s.size()[0]
    invar = th.zeros(nCells, 5).type(dtype)

    s2 = s.bmm(s)
    r2 = r.bmm(r)
    s3 = s2.bmm(s)
    r2s = r2.bmm(s)
    r2s2 = r2.bmm(s2)

    invar[:, 0] = (s2[:, 0, 0] + s2[:, 1, 1] + s2[:, 2, 2])  # Tr(s2)
    invar[:, 1] = (r2[:, 0, 0] + r2[:, 1, 1] + r2[:, 2, 2])  # Tr(r2)
    invar[:, 2] = (s3[:, 0, 0] + s3[:, 1, 1] + s3[:, 2, 2])  # Tr(s3)
    invar[:, 3] = (r2s[:, 0, 0] + r2s[:, 1, 1] + r2s[:, 2, 2])  # Tr(r2s)
    invar[:, 4] = (r2s2[:, 0, 0] + r2s2[:, 1, 1] + r2s2[:, 2, 2])  # Tr(r2s2)

    return invar


def get_invariants_fs2(s, r, grad_k):
    """
    function to compute additional invariants from Wang et al. (2018)
    :param s: normalized mean rate of strain (N x 3 x 3)
    :param r: normalized mean rate of rotation (N x 3 x 3)
    :param grad_k: normalized turbulent kinetic energy gradient (N x 3)
    :return: invar_fs2: invariants from feature set 2 (N x 13)
    """

    inv = th.zeros(s.shape[0], 13).type(dtype)
    ak = th.zeros_like(s).type(dtype)
    identity = th.eye(3).unsqueeze(0).expand(s.shape).type(dtype)

    for i in range(3):
        ak[:, :, i] = -th.cross(identity[:, :, i], grad_k)

    inv[:, 0] = (ak.matmul(ak))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)
    inv[:, 1] = (ak.matmul(ak).matmul(s))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)
    inv[:, 2] = (ak.matmul(ak).matmul(s).matmul(s))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)
    inv[:, 3] = (ak.matmul(ak).matmul(s).matmul(ak).matmul(s))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)
    inv[:, 4] = (r.matmul(ak))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)
    inv[:, 5] = (r.matmul(ak).matmul(s))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)
    inv[:, 6] = (r.matmul(ak).matmul(s).matmul(s))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)
    inv[:, 7] = (r.matmul(r).matmul(ak).matmul(s))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)
    inv[:, 8] = (ak.matmul(ak).matmul(r).matmul(s))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)
    inv[:, 9] = (r.matmul(r).matmul(ak).matmul(s).matmul(s))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)
    inv[:, 10] = (ak.matmul(ak).matmul(r).matmul(s).matmul(s))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)
    inv[:, 11] = (r.matmul(r).matmul(s).matmul(ak).matmul(s).matmul(s))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)
    inv[:, 12] = (ak.matmul(ak).matmul(s).matmul(r).matmul(s).matmul(s))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)

    return inv


def get_invariants_fs3(s, r, rs, u, grad_u, grad_p, grad_k, k, epsilon, d, nu):
    """
    function to compute additional invariants from Wang et al. (2017)
    :param s: mean rate of strain (N x 3 x 3)
    :param r: mean rate of roatation (N x 3 x 3)
    :param rs: reynolds stresses (N x 3 x 3)
    :param u: velocity (N x 3)
    :param grad_u: velocity gradient (N x 3 x 3)
    :param grad_p: pressure gradient (N x 3)
    :param grad_k: turbulent kinetic energy gradient (N x 3)
    :param k: turbulent kinetic energy (N)
    :param epsilon: energy dissipation rate (N)
    :param d: wall distance (N)
    :param nu: kinematic viscosity (scalar)
    :return: invariants: feature set 3 invariants (N x 9)
    """

    # initialize invariants
    inv = th.zeros(s.shape[0], 9).type(dtype)

    # compute s_hat, r_hat
    s_hat = (k / epsilon).unsqueeze(1).unsqueeze(2) * s
    r_hat = (k / epsilon).unsqueeze(1).unsqueeze(2) * r

    # ration of excess rotation rate to strain rate
    sst = s_hat.matmul(s_hat.transpose(1, 2))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)
    rrt = r_hat.matmul(r_hat.transpose(1, 2))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)
    inv_raw = 0.5 * (rrt - sst)
    norm = sst
    inv[:, 0] = inv_raw / (th.abs(inv_raw) + th.abs(norm))

    # turbulent intensity
    inv_raw = k
    norm = (0.5 * th.einsum('ij,ij->i', u, u))
    inv[:, 1] = inv_raw / (th.abs(inv_raw) + th.abs(norm))

    # wall-distance based Reynolds number, no norm!
    inv[:, 2] = th.min((th.sqrt(k)*d)/(50*nu), th.tensor(2.0))

    # pressure gradient along a streamline
    inv_raw = th.einsum('ij,ij->i', u, grad_p)
    norm = th.sqrt(th.einsum('ij,ij->i', grad_p, grad_p) * th.einsum('ij,ij->i', u, u))
    inv[:, 3] = inv_raw / (th.abs(inv_raw) + th.abs(norm))

    # ration of turbulent time scale to mean strain time scale
    inv_raw = k/epsilon
    # norm = 1/(th.sqrt(sst))
    norm = 1/(th.sqrt(s.matmul(s.transpose(1, 2))[:, [0, 1, 2], [0, 1, 2]].sum(axis=1)))
    inv[:, 4] = inv_raw / (th.abs(inv_raw) + th.abs(norm))

    # ratio of pressure normal stresses to shear stresses
    inv_raw = th.sqrt(th.einsum('ij,ij->i', grad_p, grad_p))
    norm = 0.5 * th.einsum('ijj,ij->i', grad_u, u)
    inv[:, 5] = inv_raw / (th.abs(inv_raw) + th.abs(norm))

    # ratio of convection to production of TKE
    inv_raw = th.einsum('ij,ij->i', u, grad_k)
    norm = th.einsum('ijk,ijk->i', rs, s)
    inv[:, 6] = inv_raw / (th.abs(inv_raw) + th.abs(norm))

    # ratio of total to normal Reynolds stresses
    inv_raw = th.sqrt(th.einsum('ijk, ijk -> i', rs, rs))
    norm = k
    inv[:, 7] = inv_raw / (th.abs(inv_raw) + th.abs(norm))

    # non-orthogonality between velocity and its gradient
    inv_raw = th.abs(th.einsum('ij, ik, ijk -> i', u, u, grad_u))
    norm = th.sqrt(th.einsum('ij,ij,ik,ikl,im,iml -> i', u, u, u, grad_u, u, grad_u))
    inv[:, 8] = inv_raw / (th.abs(inv_raw) + th.abs(norm))

    return inv


def get_tensor_functions(s, r):
    """
    function for computation of tensor basis
        Inputs:
            s: N x 3 x 3 (N is number of datapoints)
            r: N x 3 x 3
        Outputs:
            T : N x 10 x 3 x 3 (10 is number of basis tensors)
    """

    nCells = s.size()[0]
    t = th.zeros(nCells, 10, 3, 3).type(dtype)

    s2 = s.bmm(s)
    r2 = r.bmm(r)
    sr = s.bmm(r)
    rs = r.bmm(s)

    t[:, 0] = s
    t[:, 1] = sr - rs
    t[:, 2] = s2 - (1.0 / 3.0) * th.eye(3).type(dtype) \
              * (s2[:, 0, 0] + s2[:, 1, 1] + s2[:, 2, 2]).unsqueeze(1).unsqueeze(1)
    t[:, 3] = r2 - (1.0 / 3.0) * th.eye(3).type(dtype) \
              * (r2[:, 0, 0] + r2[:, 1, 1] + r2[:, 2, 2]).unsqueeze(1).unsqueeze(1)
    t[:, 4] = r.bmm(s2) - s2.bmm(r)
    t0 = s.bmm(r2)
    t[:, 5] = r2.bmm(s) + s.bmm(r2) - (2.0 / 3.0) * th.eye(3).type(dtype) \
              * (t0[:, 0, 0] + t0[:, 1, 1] + t0[:, 2, 2]).unsqueeze(1).unsqueeze(1)
    t[:, 6] = rs.bmm(r2) - r2.bmm(sr)
    t[:, 7] = sr.bmm(s2) - s2.bmm(rs)
    t0 = s2.bmm(r2)
    t[:, 8] = r2.bmm(s2) + s2.bmm(r2) - (2.0 / 3.0) * th.eye(3).type(dtype) \
              * (t0[:, 0, 0] + t0[:, 1, 1] + t0[:, 2, 2]).unsqueeze(1).unsqueeze(1)
    t[:, 9] = r.bmm(s2).bmm(r2) - r2.bmm(s2).bmm(r)

    return t


def filterField(inputData, std, filter_spatial='Gaussian'):
    """
    Filter a field (e.g. predicted b_ij) spatially using a gaussian or median filter
    """
    if len(inputData.shape) == 4:
        outputData = np.zeros(inputData.shape)
        for i1 in range(inputData.shape[0]):
            for i2 in range(inputData.shape[1]):
                if filter_spatial == 'Gaussian':
                    outputData[i1, i2, :, :] = ndimage.gaussian_filter(inputData[i1, i2, :, :],
                                                                       std, order=0, output=None, mode='nearest',
                                                                       cval=0.0, truncate=10.0)
                elif filter_spatial == 'Median':
                    outputData[i1, i2, :, :] = ndimage.median_filter(inputData[i1, i2, :, :],
                                                                     size=std, mode='nearest')

    else:  # TODO: other input shapes
        pass

    return outputData


def ph_interp(x_new):
    """
    gives back spline interpolation for points on bottom boundary
    :return: interpolation function (takes x, gives back y)
    """
    # define bottom boundary
    x = (1.929 / 54) * np.array([0., 0.1, 9., 14., 20., 30., 40., 53.9, 54.])
    x = np.append(x, 9 - x[::-1])
    y = (1.929 / 54) * np.array([28., 28., 27., 24., 19., 11., 4., 0., 0.])
    y = np.append(y, y[::-1])

    # spline interpolation
    return interp1d(x, y, kind='cubic', fill_value='extrapolate')(x_new)


def cdc_interp(x_new, re='12600'):
    """
    gives back spline interpolation for points on bottom boundary
    :return: interpolation function (takes x, gives back y)
    """
    surf_path = '/home/leonriccius/PycharmProjects/data_driven_rans/scripts/files/cdc_' + re + '.dat'
    surfacePoints = np.genfromtxt(surf_path, delimiter=' ', skip_header=3,
                                  skip_footer=0, names=["X", "Y"])

    # Seperate out the points on the bump
    bumpPoints = []
    for p0 in surfacePoints:
        if not p0['Y'] <= 10 ** (-8):
            bumpPoints.append([p0['X'], p0['Y']])
    bumpPoints = np.array(bumpPoints)
    start_point = np.zeros((1, 2))

    if int(re) == 12600:
        end_point = np.array([[12.56630039, 0.0]])
    else:
        end_point = np.array([[12.500906900, 0.0]])

    bumpPoints = np.concatenate((start_point, bumpPoints, end_point))

    return interp1d(bumpPoints[:, 0], bumpPoints[:, 1], kind='linear', fill_value='extrapolate')(x_new)


def cbfs_interp(x_new):
    """
    gives back spline interpolation for points on bottom boundary
    :return: interpolation function (takes x, gives back y)
    """
    surf_path = '/home/leonriccius/PycharmProjects/data_driven_rans/scripts/files/cbfs.dat'
    surfacePoints = np.genfromtxt(surf_path, delimiter=' ', skip_header=3,
                                  skip_footer=0, names=["X", "Y"])

    # Seperate out the points on the bump
    bumpPoints = []
    for p0 in surfacePoints:
        if not p0['Y'] <= -10 ** (-8):
            bumpPoints.append([p0['X'], p0['Y']])
    bumpPoints = np.array(bumpPoints)
    start_point = np.array([[-7.34, 1.]])
    end_point = np.array([[15.4, 0.0]])

    bumpPoints = np.concatenate((start_point, bumpPoints, end_point))

    return interp1d(bumpPoints[:, 0], bumpPoints[:, 1], kind='linear', fill_value='extrapolate')(x_new)


def mask_boundary_points(x, y, flowcase):
    """
    gives a mask that ensures all points that are blthickness far away form boundary are labelled True
    :param cutoff_y: thickness of boundarylayer to cut in y-direction dtype=float
    :param cutoff_x: thickness of boundarylayer to cut in x-direction dtype=float
    :param flowcase:
    :param x: x coordinate dtype=float
    :param y: y coordinate dytpe=float
    :param blthickness: thickness of boundarylayer to cut dtype=float
    :return: mask dtype=np.array(bool)
    """
    mask = np.ones(x.shape, dtype=bool)

    if flowcase == 'PH':
        cutoff_x, cutoff_y = 0.05, 0.02
        y_interp = ph_interp(x)
        mask[np.where(y < y_interp + cutoff_y)] = False
        mask[np.where(y > 3.035 - cutoff_y)] = False
        mask[np.where(x < 0. + cutoff_x)] = False
        mask[np.where(x > 9. - cutoff_x)] = False

    if flowcase == 'CDC12600':
        cutoff_x, cutoff_y = 0.0, 0.0
        y_interp = cdc_interp(x, re='12600')
        mask[np.where(y < y_interp + cutoff_y)] = False
        mask[np.where(y > 2. - cutoff_y)] = False
        mask[np.where(x < 0. + cutoff_x)] = False
        mask[np.where(x > 12.5663 - cutoff_x)] = False

    if flowcase == 'CDC7900':
        cutoff_x, cutoff_y = 0.0, 0.0
        y_interp = cdc_interp(x, re='7900')
        mask[np.where(y < y_interp + cutoff_y)] = False
        mask[np.where(y > 2. - cutoff_y)] = False
        mask[np.where(x < 0. + cutoff_x)] = False
        mask[np.where(x > 12.5663 - cutoff_x)] = False

    if flowcase == 'CBFS':
        cutoff_x, cutoff_y = 0.0, 0.0
        y_interp = cbfs_interp(x)
        mask[np.where(y < y_interp + cutoff_y)] = False
        mask[np.where(y > 9.51233283532541 - cutoff_y)] = False
        mask[np.where(x < -7.34 + cutoff_x)] = False
        mask[np.where(x > 15.4 - cutoff_x)] = False

    return mask


def tecplot_reader(file_path, nb_var, skiprows):
    """
    Read in Tecplot files
    :param file_path: path to tecplot file to read in
    :param nb_var: number of variables stored in file
    :param skiprows: number of rows in header to skip
    :return: tuple of arrays
    """
    arrays = []
    with open(file_path, 'r') as a:
        for idx, line in enumerate(a.readlines()):
            if idx < skiprows:
                continue
            else:
                arrays.append([float(s) for s in line.split()])
    arrays = np.concatenate(arrays)
    return np.split(arrays, nb_var)


def anisotropy(rs, k, set_nan_to_0=False):
    """calculate normalized anisotropy tensor"""

    if th.max(k == 0.0):
        print('Warning, b contains nan entries. Consider removing nan data points')

    if set_nan_to_0:
        k = th.maximum(k, th.tensor(1e-8)).unsqueeze(0).unsqueeze(0).expand(3, 3, k.size()[0]).permute(2, 0, 1)
    else:
        k = k.unsqueeze(0).unsqueeze(0).expand(3, 3, k.size()[0]).permute(2, 0, 1)

    b = rs[:] / (2 * k) - 1 / 3 * th.eye(3).unsqueeze(0).expand(k.shape[0], 3, 3)

    return b


def enforce_zero_trace(tensor):
    """
    input a set of basis tensors and get back a set of traceless basis tensors
    :param tensor: N_points x 10 x 3 x 3
    :return: tensor: N_points x 10 x 3 x 3
    """
    print('enforcing 0 trace ...')
    return tensor - 1. / 3. * th.eye(3).unsqueeze(0).unsqueeze(0).expand_as(tensor) \
           * (tensor[:, :, 0, 0] + tensor[:, :, 1, 1] + tensor[:, :, 2, 2]).unsqueeze(2).unsqueeze(3).expand_as(tensor)


def enforce_realizability(tensor, margin=0.0):
    """
    input a set of anisotropy tensors and get back a set of tensors that does not violate realizabiliy constraints.
    creates labels for branchless if clause. labels hold true where realizability contraints are violated and corrects
    entries
    :param margin: (float) set to small value larger than 0 to push eigenvalues over -1/3 instead
    asymptotically approach the boundary
    :param tensor: N_points x 3 x 3
    :return: N_points x 3 x 3
    """
    # ensure b_ii > -1/3
    diag_min = th.min(tensor[:, [0, 1, 2], [0, 1, 2]], 1)[0].unsqueeze(1)
    labels = (diag_min < th.tensor(-1. / 3.))
    tensor[:, [0, 1, 2], [0, 1, 2]] *= labels * (-1. / (3. * diag_min)) + ~labels

    # ensure 2*|b_ij| < b_ii + b_jj + 2/3
    # b_12
    labels = (2 * th.abs(tensor[:, 0, 1]) > tensor[:, 0, 0] + tensor[:, 1, 1] + 2. / 3.).unsqueeze(1)
    tensor[:, [0, 1], [1, 0]] = labels * (tensor[:, 0, 0] + tensor[:, 1, 1] + 2. / 3.).unsqueeze(1) \
                                * .5 * th.sign(tensor[:, [0, 1], [1, 0]]) + ~labels * tensor[:, [0, 1], [1, 0]]

    # b_23
    labels = (2 * th.abs(tensor[:, 1, 2]) > tensor[:, 1, 1] + tensor[:, 2, 2] + 2. / 3.).unsqueeze(1)
    tensor[:, [1, 2], [2, 1]] = labels * (tensor[:, 1, 1] + tensor[:, 2, 2] + 2. / 3.).unsqueeze(1) \
                                * .5 * th.sign(tensor[:, [1, 2], [2, 1]]) + ~labels * tensor[:, [1, 2], [2, 1]]

    # b_13
    labels = (2 * th.abs(tensor[:, 0, 2]) > tensor[:, 0, 0] + tensor[:, 2, 2] + 2. / 3.).unsqueeze(1)
    tensor[:, [0, 2], [2, 0]] = labels * (tensor[:, 0, 0] + tensor[:, 2, 2] + 2. / 3.).unsqueeze(1) \
                                * .5 * th.sign(tensor[:, [0, 2], [2, 0]]) + ~labels * tensor[:, [0, 2], [2, 0]]

    # ensure positive semidefinitness by pushing smallest eigenvalue to -1/3. Reynolds stress eigenvalues are then
    # positive semidefinite
    # ensure lambda_1 > (3*|lambda_2| - lambda_2)/2
    eigval, eigvec = th.symeig(tensor, eigenvectors=True)
    labels = (eigval[:, 2] < (3 * th.abs(eigval[:, 1]) - eigval[:, 1]) * .5).unsqueeze(1)
    eigval *= labels * ((3. * th.abs(eigval[:, 1]) - eigval[:, 1]) / (2. * eigval[:, 2])).unsqueeze(1) + ~labels
    print('Violation of condition 1: {}'.format(th.max(labels)))

    # ensure lambda_1 < 1/3 - lambda_2
    labels = (eigval[:, 2] > 1. / 3. - eigval[:, 1]).unsqueeze(1)
    eigval *= labels * ((1. / 3. - eigval[:, 1]) / (eigval[:, 2]) - margin).unsqueeze(1) + ~labels
    print('Violation of condition 2: {}'.format(th.max(labels)))

    # rotate tensor back to initial frame
    tensor = eigvec.matmul(th.diag_embed(eigval).matmul(eigvec.transpose(1, 2)))

    return tensor


def load_standardized_data(data):
    """
    function to load in all fluid data, convert it to torch tensors, and interpolate dns data on rans grid
    :param data: dict with different flow geometries
    :return:
    """

    assert (os.path.exists(data['home'])), 'dictionary specified as \'home\' does not exist'

    for i, case in enumerate(data['flowCase']):
        for j, re in enumerate(data['Re'][i]):
            print((case, re))
            # define dns, rans, and target paths
            dns_path = os.sep.join([data['home'], data['dns'], case, re, 'tensordata'])
            rans_path = os.sep.join(
                [data['home'], data['rans'], case, 'Re{}_{}_{}'.format(re, data['model'][i][j], data['ny'][i][j])])
            rans_time = data['ransTime'][i][j]
            target_path = os.sep.join([data['home'], data['target_dir'], case, re])
            print(target_path)

            # load dns data
            grid_dns = th.load(os.sep.join([dns_path, 'grid-torch.th']))
            b_dns = th.load(os.sep.join([dns_path, 'b-torch.th']))
            rs_dns = th.load(os.sep.join([dns_path, 'rs-torch.th']))
            k_dns = th.load(os.sep.join([dns_path, 'k-torch.th']))
            u_dns = th.load(os.sep.join([dns_path, 'u-torch.th']))
            if os.path.isfile(os.sep.join([dns_path, 'p-torch.th'])):
                is_p = True
                p_dns = th.load(os.sep.join([dns_path, 'p-torch.th']))
            else:
                is_p = False

            print('rans time:  ', rans_time)

            # load rans data (grid, u, k, rs, epsilon, grad_u, grad_k, grad_p, yWall)
            grid_rans = pre.readCellCenters(rans_time, rans_path)
            u_rans = pre.readVectorData(rans_time, 'U', rans_path)
            k_rans = pre.readScalarData(rans_time, 'k', rans_path)
            if os.path.isfile(os.sep.join([rans_path, rans_time, 'turbulenceProperties:R'])):
                rs_rans = pre.readSymTensorData(rans_time, 'turbulenceProperties:R', rans_path).reshape(-1, 3, 3)
            else:
                rs_rans = pre.readSymTensorData(rans_time, 'R', rans_path).reshape(-1, 3, 3)
            if os.path.isfile(os.sep.join([rans_path, rans_time, 'gradU'])):
                grad_u_rans = pre.readTensorData(rans_time, 'gradU', rans_path)  # or 'gradU
            else:
                grad_u_rans = pre.readTensorData(rans_time, 'grad(U)', rans_path)
            grad_k_rans = pre.readVectorData(rans_time, 'grad(k)', rans_path)
            grad_p_rans = pre.readVectorData(rans_time, 'grad(p)', rans_path)

            if os.path.isfile(os.sep.join([rans_path, rans_time, 'yWall'])):
                y_wall_rans = pre.readScalarData(rans_time, 'yWall', rans_path)
            else:
                y_wall_rans = pre.readScalarData(rans_time, 'wallDistance', rans_path)
            # else:
            #     print("Error: No yWall or wallDistance in {} {}".format(case, re))
            #     break

            # reading in epsilon, otherwise calculate from omega
            if os.path.isfile(os.sep.join([rans_path, rans_time, 'epsilon'])):
                epsilon_rans = pre.readScalarData(rans_time, 'epsilon', rans_path)
            else:
                omega_rans = pre.readScalarData(rans_time, 'omega', rans_path)  # 'epsilon' or 'omega'
                epsilon_rans = omega_rans * k_rans * 0.09  # 0.09 is beta star

            # get nu from dict
            nu = data['nu'][i][j]

            # calculate mean rate of strain and rotation tensors
            s = 0.5 * (grad_u_rans + grad_u_rans.transpose(1, 2))
            r = 0.5 * (grad_u_rans - grad_u_rans.transpose(1, 2))

            # normalize s, r, grad_k
            s_hat = (k_rans / epsilon_rans).unsqueeze(1).unsqueeze(2) * s
            r_hat = (k_rans / epsilon_rans).unsqueeze(1).unsqueeze(2) * r
            grad_k_hat = (th.sqrt(k_rans) / epsilon_rans).unsqueeze(1) * grad_k_rans

            # cap s and r tensors
            if data['capSandR']:
                if th.max(th.abs(s_hat)) > 6. or th.max(th.abs(r_hat)) > 6.:
                    print('capping tensors ...')
                    s_hat = cap_tensor(s_hat, 6.0)
                    r_hat = cap_tensor(r_hat, 6.0)

            # calculate invariants
            inv = get_invariants(s_hat, r_hat)
            t = get_tensor_functions(s_hat, r_hat)

            # correct invariants of cases with symmetries
            if data['correctInvariants']:
                print('Setting invariants 3 and 4 to 0 ...')
                inv[:, [2, 3]] = 0.0

            # compute invariants and exclude features
            # FS1
            if 'FS1' in data:
                inv_fs1 = get_invariants(s_hat, r_hat)

                if data['FS1']['excludeFeatures']:
                    for k in data['FS1']['features'][::-1]:  # must be in reverse of indices are wrong
                        inv_fs1 = th.cat((inv_fs1[:, :k - 1], inv_fs1[:, k:]), dim=1)

            # FS2
            if 'FS2' in data:
                inv_fs2 = get_invariants_fs2(s_hat, r_hat, grad_k_hat)

                if data['FS2']['excludeFeatures']:
                    for k in data['FS2']['features'][::-1]:  # must be in reverse of indices are wrong
                        inv_fs2 = th.cat((inv_fs2[:, :k - 1], inv_fs2[:, k:]), dim=1)

            # FS3
            if 'FS3' in data:
                inv_fs3 = get_invariants_fs3(s, r, rs_rans, u_rans, grad_u_rans, grad_p_rans,
                                             grad_k_rans, k_rans, epsilon_rans, y_wall_rans, nu)

                if data['FS3']['excludeFeatures']:
                    for k in data['FS3']['features'][::-1]:  # must be in reverse of indices are wrong
                        inv_fs3 = th.cat((inv_fs3[:, :k - 1], inv_fs3[:, k:]), dim=1)

            # compute tensorbasis and enforce zero trace
            t = get_tensor_functions(s_hat, r_hat)
            if data['enforceZeroTrace']:
                t = enforce_zero_trace(t)

            # compute anisotropy tensor b
            b_rans = anisotropy(rs_rans, k_rans, data['removeNan'])

            # interpolate dns data on rans grid
            print('Interpolating DNS data on RANS grid ...')
            method = data['interpolationMethod']
            if case == 'SquareDuct':
                int_grid = grid_rans[:, [1, 2, 0]]
            else:
                int_grid = grid_rans

            b_dns_interp = th.tensor(griddata(grid_dns[:, 0:2], b_dns, (int_grid[:, 0], int_grid[:, 1]),
                                              method=method))
            u_dns_interp = th.tensor(griddata(grid_dns[:, 0:2], u_dns, (int_grid[:, 0], int_grid[:, 1]),
                                              method=method))
            rs_dns_interp = th.tensor(griddata(grid_dns[:, 0:2], rs_dns, (int_grid[:, 0], int_grid[:, 1]),
                                               method=method))
            k_dns_interp = th.tensor(griddata(grid_dns[:, 0:2], k_dns, (int_grid[:, 0], int_grid[:, 1]),
                                              method=method))
            if is_p:
                p_dns_interp = th.tensor(griddata(grid_dns[:, 0:2], p_dns, (int_grid[:, 0], int_grid[:, 1]),
                                                  method=method))

            if data['saveTensors']:
                # create directories if not defined
                if not os.path.exists(target_path):
                    os.makedirs(target_path)

                # save dns
                th.save(b_dns_interp, os.sep.join([target_path, 'b_dns-torch.th']))
                th.save(rs_dns_interp, os.sep.join([target_path, 'rs_dns-torch.th']))
                th.save(k_dns_interp, os.sep.join([target_path, 'k_dns-torch.th']))
                th.save(u_dns_interp, os.sep.join([target_path, 'u_dns-torch.th']))
                if is_p:
                    th.save(p_dns_interp, os.sep.join([target_path, 'p_dns-torch.th']))

                # save rans
                th.save(grid_rans, os.sep.join([target_path, 'grid-torch.th']))
                th.save(u_rans, os.sep.join([target_path, 'u_rans-torch.th']))
                th.save(rs_rans, os.sep.join([target_path, 'rs_rans-torch.th']))
                th.save(b_rans, os.sep.join([target_path, 'b_rans-torch.th']))
                th.save(k_rans, os.sep.join([target_path, 'k_rans-torch.th']))
                th.save(inv, os.sep.join([target_path, 'inv-torch.th']))
                th.save(t, os.sep.join([target_path, 't-torch.th']))
                th.save(inv_fs1, os.sep.join([target_path, 'inv_fs1-torch.th']))
                th.save(inv_fs2, os.sep.join([target_path, 'inv_fs2-torch.th']))
                th.save(inv_fs3, os.sep.join([target_path, 'inv_fs3-torch.th']))

    return 0


import matplotlib.pyplot as plt

if __name__ == '__main__':

    # initialize dictionary
    data = {}

    # set paths
    data['home'] = '/home/leonriccius/Documents/Fluid_Data'
    data['dns'] = 'dns'
    data['rans'] = 'rans'
    data['target_dir'] = 'tensordata'

    # set options for data loading
    data['FS1'] = {'excludeFeatures': True,
                   'features': [3, 4]}
    data['FS2'] = {'excludeFeatures': True,
                   'features': [4, 5, 6, 7, 8, 10, 11, 12]}
    data['FS3'] = {'excludeFeatures': False}
    data['interpolationMethod'] = 'linear'
    data['enforceZeroTrace'] = True
    data['capSandR'] = True
    data['saveTensors'] = True
    data['removeNan'] = True
    data['correctInvariants'] = True

    # set flow cases to load
    data['flowCase'] = ['CurvedBackwardFacingStep']
    data['Re'] = [['13700']]
    data['nu'] = [[7.299270072992701e-05]]
    data['nx'] = [[300]]
    data['ny'] = [[150]]
    data['model'] = [['kOmega']]
    data['ransTime'] = [['7000']]

    load_standardized_data(data)

    # # for testing stored invariants
    # inv_fs1 = th.load(
    #     '/home/leonriccius/Documents/Fluid_Data/tensordata_fs1_fs2_fs3_reduced/ConvDivChannel/12600/inv_fs1-torch.th')
    # inv_fs2 = th.load(
    #     '/home/leonriccius/Documents/Fluid_Data/tensordata_fs1_fs2_fs3_reduced/ConvDivChannel/12600/inv_fs2-torch.th')
    # inv_fs3 = th.load(
    #     '/home/leonriccius/Documents/Fluid_Data/tensordata_fs1_fs2_fs3_reduced/ConvDivChannel/12600/inv_fs3-torch.th')
    # grid = th.load(
    #     '/home/leonriccius/Documents/Fluid_Data/tensordata_fs1_fs2_fs3_reduced/ConvDivChannel/12600/grid-torch.th')



    # # for testing invariants from raw data
    # rans_path = '/home/leonriccius/Documents/Fluid_Data/rans_kaandorp/PeriodicHills/Re10595_kOmega_150'
    # rans_time = '30000'
    #
    # u = pre.readVectorData(rans_time, 'U', rans_path)
    # grid = pre.readCellCenters(rans_time, rans_path)
    # rs = pre.readSymTensorData(rans_time, 'turbulenceProperties:R', rans_path).reshape(-1, 3, 3)
    # grad_u = pre.readTensorData(rans_time, 'grad(U)', rans_path)
    # grad_k = pre.readVectorData(rans_time, 'grad(k)', rans_path)
    # grad_p = pre.readVectorData(rans_time, 'grad(p)', rans_path)
    # yWall = pre.readScalarData(rans_time, 'yWall', rans_path)
    # k = pre.readScalarData(rans_time, 'k', rans_path)
    # omega = pre.readScalarData(rans_time, 'omega', rans_path)  # 'epsilon' or 'omega'
    # epsilon = omega * k * 0.09  # 0.09 is beta star
    # nu = 9.438414346389807e-05
    #
    # # calculate mean rate of strain and rotation tensors
    # s = 0.5 * (grad_u + grad_u.transpose(1, 2))
    # r = 0.5 * (grad_u - grad_u.transpose(1, 2))
    #
    # # normalize s and r
    # s_hat = (k / epsilon).unsqueeze(1).unsqueeze(2) * s
    # r_hat = (k / epsilon).unsqueeze(1).unsqueeze(2) * r
    # grad_k_hat = (th.sqrt(k) / epsilon).unsqueeze(1) * grad_k
    #
    # inv = get_invariants_fs3(s, r, rs, u, grad_u, grad_p, grad_k, k, epsilon, yWall, nu)


    # # for plotting invariants
    # import matplotlib
    #
    # cmap = matplotlib.cm.get_cmap("coolwarm")
    # #
    # inv = inv_fs2
    # for i in range(5):
    #     inv_min = th.min(inv[:, i])
    #     inv_max = th.max(inv[:, i])
    #     levels = np.linspace(inv_min, inv_max, 50)
    #     fig, ax = plt.subplots(figsize=(9, 3))
    #     plot = ax.tricontourf(grid[:, 0], grid[:, 1], inv[:, i], cmap=cmap, levels=levels)
    #     ax.set_title('Inv {}'.format(i + 1))
    #     fig.colorbar(plot)
    #     fig.show()


    # # data from all flow cases to test load_data function
    # data['flowCase'] = ['PeriodicHills']
    # data['Re'] = [['10595']]
    # data['nx'] = [[140]]
    # data['ny'] = [[150]]
    # data['model'] = [['kOmega']]
    # data['ransTime'] = [['30000']]
    # data['nu'] = [[9.438414346389807e-05]]

    # # set flow cases to load
    # data['flowCase'] = ['PeriodicHills',
    #                     'ConvDivChannel',
    #                     'CurvedBackwardFacingStep',
    #                     'SquareDuct']
    # data['Re'] = [['700', '1400', '2800', '5600', '10595'],
    #               ['12600', '7900'],
    #               ['13700'],
    #               ['1800', '2000', '2400', '2600', '2900', '3200', '3500']]
    # data['nu'] = [[1.4285714285714286e-03, 7.142857142857143e-04, 3.5714285714285714e-04, 1.7857142857142857e-04,
    #                9.438414346389807e-05],
    #               [7.936507936507937e-05, 1.26582e-04],
    #               [7.299270072992701e-05],
    #               [0.00026776, 0.00024098, 0.00020083, 0.00018537, 0.00016619, 0.00015061, 0.00013770]]
    # data['nx'] = [[140, 140, 140, 140, 140],
    #               [140, 140],
    #               [140],
    #               [50, 50, 50, 50, 50, 50, 50]]
    # data['ny'] = [[150, 150, 150, 150, 150],
    #               [100, 100],
    #               [150],
    #               [50, 50, 50, 50, 50, 50, 50]]
    # data['model'] = [['kOmega', 'kOmega', 'kOmega', 'kOmega', 'kOmega'],
    #                  ['kOmega', 'kOmega'],
    #                  ['kOmega'],
    #                  ['kOmega', 'kOmega', 'kOmega', 'kOmega', 'kOmega', 'kOmega', 'kOmega']]
    # data['ransTime'] = [['30000', '30000', '30000', '30000', '30000'],
    #                     ['7000', '7000'],
    #                     ['3000'],
    #                     ['40000', '40000', '50000', '50000', '50000', '50000', '50000']]

    # # for reloading square duct
    # data['flowCase'] = ['SquareDuct']
    # data['Re'] = [['1800', '2000', '2400', '2600', '2900', '3200', '3500']]
    # data['nu'] = [[0.00026776, 0.00024098, 0.00020083, 0.00018537, 0.00016619, 0.00015061, 0.00013770]]
    # data['nx'] = [[50, 50, 50, 50, 50, 50, 50]]
    # data['ny'] = [[50, 50, 50, 50, 50, 50, 50]]
    # data['model'] = [['kOmega', 'kOmega', 'kOmega', 'kOmega', 'kOmega', 'kOmega', 'kOmega']]
    # data['ransTime'] = [['40000', '40000', '50000', '50000', '50000', '50000', '50000']]
