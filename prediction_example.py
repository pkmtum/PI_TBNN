from scripts import barymap
from scripts.utilities import *
from scripts.torchToFoam import writesymmtensor

# standard datatype
dtype = th.double

if __name__ == '__main__':

    # set path to models and initialise arrays
    model_path = './storage/models/example_case/0e+00'
    model = th.load(os.sep.join([model_path, 'model.pt']))
    mu = th.load(os.sep.join([model_path, 'mu.th']))
    std = th.load(os.sep.join([model_path, 'std.th']))
    _ = model.eval()  # mandatory, see torch.load doc

    # read in tensor data
    rans_path = './storage/Fluid_Data/rans/ConvDivChannel/Re7900_kOmega_100_ml/'
    # pred_path = '/home/leonriccius/Documents/Fluid_Data/rans_kaandorp/PeriodicHills/Re5600_kOmega_150/'

    rans_time = max([int(entry) for entry in os.listdir(rans_path) if entry.isnumeric()])
    rans_time = 7000

    with NoStdStreams():
        u = pre.readVectorData(rans_time, 'U', rans_path)
        grid = pre.readCellCenters(rans_time, rans_path)
        rs = pre.readSymTensorData(rans_time, 'turbulenceProperties:R', rans_path).reshape(-1, 3, 3)
        grad_u = pre.readTensorData(rans_time, 'grad(U)', rans_path)
        grad_k = pre.readVectorData(rans_time, 'grad(k)', rans_path)
        grad_p = pre.readVectorData(rans_time, 'grad(p)', rans_path)
        y_wall = pre.readScalarData(rans_time, 'wallDistance', rans_path)
        k = pre.readScalarData(rans_time, 'k', rans_path)
        omega = pre.readScalarData(rans_time, 'omega', rans_path)  # 'epsilon' or 'omega'
        epsilon = omega * k * 0.09  # 0.09 is beta star
        nu = 3.5714285714285714e-04

    # get b_rans
    b_rans = barymap.anisotropy(rs, k)

    # calculate mean rate of strain and rotation tensors
    s = 0.5 * (grad_u + grad_u.transpose(1, 2))
    r = 0.5 * (grad_u - grad_u.transpose(1, 2))

    # normalize s and r
    s_hat = (k / epsilon).unsqueeze(1).unsqueeze(2) * s
    r_hat = (k / epsilon).unsqueeze(1).unsqueeze(2) * r
    grad_k_hat = (th.sqrt(k) / epsilon).unsqueeze(1) * grad_k

    # capping tensors
    s_hat = cap_tensor(s_hat, 6.0)
    r_hat = cap_tensor(r_hat, 6.0)

    # get invariants and remove features
    inv_fs1 = get_invariants(s_hat, r_hat)
    for i in [3,4][::-1]:  # must be in reverse of indices are wrong
        inv_fs1 = th.cat((inv_fs1[:, :i - 1], inv_fs1[:, i:]), dim=1)

    inv_fs2 = get_invariants_fs2(s_hat, r_hat, grad_k_hat)
    for i in [4, 5, 6, 7, 8, 10, 11, 12][::-1]:  # must be in reverse of indices are wrong
        inv_fs2 = th.cat((inv_fs2[:, :i - 1], inv_fs2[:, i:]), dim=1)

    inv_fs3 = get_invariants_fs3(s, r, rs, u, grad_u, grad_p, grad_k, k, epsilon, y_wall, nu)

    inv = th.cat((inv_fs1, inv_fs2, inv_fs3), dim=1)

    # scale invariants
    inv_mean_std = mean_std_scaling(inv, mu=mu, std=std)[0]

    # get tensor basis
    t = get_tensor_functions(s_hat, r_hat)
    t_zero_trace = enforce_zero_trace(t.reshape(-1, 10, 3, 3))

    # get prediction
    b_pred, _ = model(inv_mean_std, t_zero_trace.reshape(-1, 10, 9))
    b_pred = b_pred.reshape(-1, 3, 3).detach()

    # dns data
    path_dns = './storage/Fluid_Data/tensordata/ConvDivChannel/7900'
    b_dns = th.load(os.sep.join([path_dns, 'b_dns-torch.th']))
    grid_dns = th.load(os.sep.join([path_dns, 'grid-torch.th']))

    # enforce realizability on b (multiple interations are needed as fulfilling one contraint might violate another)
    b_real = b_pred.clone()
    for i in range(5):
        b_real = enforce_realizability(b_real)

    # output maximum b values (gives an idea if b is realizable)
    print('\nmax b:       {:.4f}'.format(th.max(b_pred[:, 0, 0])))
    print('max b_real:  {:.4f}'.format(th.max(b_real[:, 0, 0])))

    # filter b_pred
    nx, ny = 140, 100
    b_real_reshaped = b_real.reshape(ny, nx, 3, 3).permute(2, 3, 0, 1).detach().numpy()
    b_real_filt = th.from_numpy(filterField(b_real_reshaped, std=2)).reshape(3, 3, -1).permute(2, 0, 1)

    display_results = False

    if display_results:

        # define refinement level of interpolation grid and how much of boundary should be but off
        ref = 15
        gamma = 0.995

        # find min an max of x coordinate
        x_min, x_max = np.min(grid[:, 0].numpy()), np.max(grid[:, 0].numpy())
        x_mean = 0.5 * (x_max - x_min)
        x_min = x_mean - gamma * (x_mean - x_min)
        x_max = x_mean - gamma * (x_mean - x_max)

        # find min an max of y coordinate
        y_min, y_max = np.min(grid[:, 1].numpy()), np.max(grid[:, 1].numpy())
        y_mean = 0.5 * (y_max - y_min)
        y_min = y_mean - gamma * (y_mean - y_min)
        y_max = y_mean - gamma * (y_mean - y_max)

        # compute number of points per coordinate
        np_x = int(ref * (x_max - x_min))
        np_y = int(ref * (y_max - y_min))

        # get grid and shift points on curved boundaries
        grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, np_x), np.linspace(y_min, y_max, np_y))
        grid_y_shifted = cdc_interp(grid_x, re='7900') * (y_max - grid_y) / y_max + grid_y
        grid_y = grid_y_shifted

        # interpolate b tensor
        grid_b_rans = griddata(grid[:, 0:2], b_rans.numpy(), (grid_x, grid_y), method='linear')  # , fill_value=0.)
        grid_b_pred = griddata(grid[:, 0:2], b_pred.numpy(), (grid_x, grid_y), method='linear')
        grid_b_real = griddata(grid[:, 0:2], b_real.numpy(), (grid_x, grid_y), method='linear')
        grid_b_real_filt = griddata(grid[:, 0:2], b_real_filt.numpy(), (grid_x, grid_y), method='linear')
        grid_b_dns = griddata(grid_dns[:, 0:2], b_dns.numpy(), (grid_x, grid_y), method='linear')

        # get min an max of (b_pred, b_test) for colormap
        tmp = np.vstack((grid_b_rans, grid_b_real, grid_b_dns))
        b_min = np.min(tmp, axis=(0, 1))
        b_max = np.max(tmp, axis=(0, 1))

        # set plot layout
        components = [[0, 0], [1, 1], [2, 2], [0, 1]]
        ncols = len(components)
        nrows = 4
        s = 1.

        # # create figure
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(s * 3. * ncols, s * 1.25 * nrows), sharex=True,
                               sharey=True,
                               constrained_layout=True)  # set factor for ncols to 4.5 for true scaling of physical domain

        # loop over components
        for i, cmp in enumerate(components):

            # set levels for contour plots
            levels = np.linspace(b_min[cmp[0], cmp[1]], b_max[cmp[0], cmp[1]], 50)

            # plot contours
            b_rans_plot = ax[0, i].contourf(grid_x, grid_y_shifted, grid_b_rans[:, :, cmp[0], cmp[1]], levels=levels)
            b_pred_plot = ax[1, i].contourf(grid_x, grid_y_shifted, grid_b_pred[:, :, cmp[0], cmp[1]], levels=levels)
            b_real_filt_plot = ax[2, i].contourf(grid_x, grid_y_shifted, grid_b_real_filt[:, :, cmp[0], cmp[1]],
                                                 levels=levels)
            b_dns_plot = ax[3, i].contourf(grid_x, grid_y_shifted, grid_b_dns[:, :, cmp[0], cmp[1]], levels=levels)

            # remove contour lines
            for contour in [b_rans_plot, b_pred_plot, b_real_filt_plot, b_dns_plot]:
                for c in contour.collections:
                    c.set_edgecolor("face")
                    c.set_linewidth(0.00000000000000001)

            # create colorbar
            fig.colorbar(b_dns_plot, ax=ax[:, i].flat, aspect=2.5 * ncols * nrows, format='%.2f', pad=0.05)

        # create column labels
        ax[0, 0].set_title(r'$b_{11}$', fontsize='medium')
        ax[0, 1].set_title(r'$b_{22}$', fontsize='medium')
        ax[0, 2].set_title(r'$b_{33}$', fontsize='medium')
        ax[0, 3].set_title(r'$b_{12}$', fontsize='medium')

        # create row labels
        x_off = -5.0
        y_off = 1.0

        ax[0, 0].text(x_off, y_off, r'LEVM', fontsize='medium', ha='left')
        ax[1, 0].text(x_off, y_off, r'TBNN' '\n' r'raw', fontsize='medium', ha='left', va='center')
        ax[2, 0].text(x_off, y_off, r'TBNN' '\n' r'filtered', fontsize='medium', ha='left', va='center')
        ax[3, 0].text(x_off, y_off, r'DNS', fontsize='medium', ha='left', va='center')

        # show figure
        fig.show()

        # # save fig
        # fig_path = '/home/....'
        # fig_name = time() + 'cdc_7900_ml.pdf'
        # plt.savefig(os.sep.join([fig_path, fig_name]), format='pdf')

    # extract b at inflow boundary for dirichlet BC
    b_boundary = b_real_filt.reshape(ny, nx, 3, 3)[:, 0]

    # set boundary list
    b_list = [('topWall', 'fixedValue_uniform'),
              ('bottomWall', 'fixedValue_uniform'),
              ('inlet', 'fixedValue_nonuniform', b_boundary),
              ('outlet', 'zeroGradient'),
              ('frontAndBack', 'empty')]

    # set path for openfoam flow case with ml prediction and write anisotropy tensor
    dd_case_path = './storage/Fluid_Data/rans/ConvDivChannel/Re7900_kOmega_100_ml'
    print('Writing anisotropy tensor to file ...')
    writesymmtensor(b_real_filt, os.sep.join([dd_case_path, str(rans_time), 'b_ml']), b_list)
    print('Successfully wrote anisotropy tensor to ' + dd_case_path)

    # now run openfoam case with ml prediction for the anisotropy tensor
