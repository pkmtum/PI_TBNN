import numpy as np
import torch as th
import os.path
# sys.path.insert(1, '/home/leonriccius/Documents/jupyter_notebook')
import scripts.preProcess as pre

if __name__ == '__main__':

    # setting directory structure
    rans_dir =  ['Re2000_kOmega_50', 'Re1800_kOmega_50', 'Re2200_kOmega_50', 'Re2400_kOmega_50', 'Re2600_kOmega_50',
                 'Re2900_kOmega_50', 'Re3200_kOmega_50', 'Re3500_kOmega_50']
    rans_path = '/home/leonriccius/Documents/Fluid_Data/rans_kaandorp/SquareDuct'

    for i in rans_dir:

        # set directory path and extract rans time
        curr_dir = os.sep.join([rans_path, i])  # rans_dir[3]
        rans_time = max([entry for entry in os.listdir(curr_dir) if entry.isnumeric()])

        # read in cell centers, check if AE or BE of cellCenters
        cell_centers = pre.readCellCenters(rans_time, curr_dir)
        print('Succesfully read {} data points'.format(cell_centers.shape[0]))

        # check if kEpsilon or kOmega
        is_epsilon = os.path.isfile(os.sep.join([curr_dir, rans_time, 'epsilon']))

        # reading in RS
        if os.path.isfile(os.sep.join([curr_dir, rans_time, 'turbulenceProperties:R'])):
            RS = pre.readSymTensorData(rans_time, 'turbulenceProperties:R', curr_dir).reshape(-1, 3, 3)
        else:
            RS = pre.readSymTensorData(rans_time, 'R', curr_dir).reshape(-1, 3, 3)

        # reading in grad U
        if os.path.isfile(os.sep.join([curr_dir, rans_time, 'gradU'])):
            grad_U = pre.readTensorData(rans_time, 'gradU', curr_dir) # or 'gradU
        else:
            grad_U = pre.readTensorData(rans_time, 'grad(U)', curr_dir)

        # reading in k, U
        k = pre.readScalarData(rans_time, 'k', curr_dir)
        U = pre.readVectorData(rans_time, 'U', curr_dir)

        # reading in epsilon, otherwise calculate from omega
        if is_epsilon:
            epsilon = pre.readScalarData(rans_time, 'epsilon', curr_dir)
        else:
            omega = pre.readScalarData(rans_time, 'omega', curr_dir) # 'epsilon' or 'omega'
            epsilon = omega*k*0.09  # 0.09 is beta star

        # calculating S and R from velocity gradient
        S = 0.5 * (grad_U + grad_U.transpose(1, 2))
        R = 0.5 * (grad_U - grad_U.transpose(1, 2))

        # saving sliced fields
        pre.saveTensor(RS, 'RS', rans_time, curr_dir)
        pre.saveTensor(U, 'U', rans_time, curr_dir)
        pre.saveTensor(k, 'k', rans_time, curr_dir)
        pre.saveTensor(S, 'S', rans_time, curr_dir)
        pre.saveTensor(R, 'R', rans_time, curr_dir)
        pre.saveTensor(epsilon, 'epsilon', rans_time, curr_dir)
        # else:
        #     pre.saveTensor(omega, 'omega', rans_time, curr_dir)
        # saving cell center coordinates
        pre.saveTensor(cell_centers, 'grid', rans_time, curr_dir)
