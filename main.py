from scripts import TensorBasedNN
from scripts.utilities import load_standardized_data
import torch as th
import torch.nn as nn
import os
import pandas as pd

if __name__ == '__main__':

    # load data from dns/rans folders
    reload_data = False
    if reload_data:
        # initialize dictionary
        data = {}

        # set paths
        data['home'] = './storage/Fluid_Data'
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
        data['flowCase'] = ['ConvDivChannel']
        data['Re'] = [['12600', '7900']]
        data['nu'] = [[7.936507936507937e-05, 1.26582e-04]]
        data['nx'] = [[140, 140]]
        data['ny'] = [[100, 100]]
        data['model'] = [['kOmega', 'kOmega']]
        data['ransTime'] = [['7000', '7000']]

        # load in data
        load_standardized_data(data)

        print('Data successfully reloaded, start initializing TBNN ...')
        print('______________________________________________________________________________________________')
    else:
        print('Data reloading not specified, start initializing TBNN ...')
        print('______________________________________________________________________________________________')

    # set training directories and flow cases
    training_dir = './storage/Fluid_Data/tensordata'
    training_flow_geom = ['ConvDivChannel']
    training_cases = [['12600']]
    training_dirs = [os.sep.join([training_dir, geom]) for geom in training_flow_geom]

    # set array that holds weight decay parameters
    # weight_decay = 10. ** th.linspace(-14, 0, 8)
    weight_decay = [0.0]
    print('Weight decay parameter vector:\n', weight_decay)

    for weight_decay_ in weight_decay:

        # set training features and network architecture
        features = ['FS1', 'FS2', 'FS3']
        n_features = 17  # fs_1: 3,  fs_2: 5,  fs:3: 9   add number of features per set together
        geneva_architecture = [n_features, 200, 200, 200, 40, 20, 10]
        ling_architecture = [n_features, 30, 30, 30, 30, 30, 30, 30, 30, 10]
        layers = ling_architecture

        # set seed and initialize model, comment second line two use random seed
        seed = 12345
        th.manual_seed(seed)
        model = TensorBasedNN.TBNNModel(layersizes=layers,
                                        activation=nn.LeakyReLU(),
                                        final_layer_activation=nn.Identity())

        # # if pretrained model should be used
        # model.net = th.load(model_path)
        print('TBNN successfully initialized')
        print('______________________________________________________________________________________________')

        # specify path to save trained model. base path is combined with weight decay parameter choice
        base_path = ('./storage/models/example_case')
        save_model_path = os.sep.join([base_path, '{:.0e}'.format(weight_decay_)])
        # save_model_path = base_path

        # create directory if not existing yet
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        # set to True if model should be saved
        save_model = True

        # # display net modules
        # for m in model.net.modules():
        #     print(m)

        # load fluid data and normalize features
        model.load_data(training_dirs, training_cases, features, n_samples=14000)
        model.normalize_features(cap=2.)
        batch_size = 100
        model.batch_size = batch_size  # need to specify it here because select train data needs it

        # select data for training
        model.select_training_data(train_ratio=0.7, seed=seed)

        # set learning rate and scheduler
        lr_scheduler = th.optim.lr_scheduler.MultiStepLR
        lr_scheduler_dict = {'milestones': [50, 100, 150, 200, 300, 400, 500, 600, 700], 'gamma': .5, 'verbose': False}
        learning_rate = 2.5e-5

        # parameters for model training
        training_params = {'lr_initial': learning_rate, 'n_epochs': 1000, 'batch_size': batch_size,
                           'lr_scheduler': lr_scheduler, 'lr_scheduler_dict': lr_scheduler_dict,
                           'fixed_seed': True,
                           'early_stopping': True, 'moving_average': 5, 'min_epochs': 200,
                           'weight_decay': 0, 'builtin_weightdecay': False,
                           'lambda_real': weight_decay_,
                           'error_method': 'b_unique',
                           'train_set_size': model.inv_train.shape[0]}

        # train model
        model.train_model(**training_params)

        # save last validation loss and print training parameters
        training_params['last_val_loss'] = model.val_loss_vector[-1]
        print(training_params)

        # Save trained model and all relevant data
        print('Saving model and relevant data ...')
        if save_model:
            th.save(model.net, os.sep.join([save_model_path, 'model.pt']))
            th.save(model.net.state_dict(), os.sep.join([save_model_path, 'state_dict.pt']))
            th.save(model.mu, os.sep.join([save_model_path, 'mu.th']))
            th.save(model.std, os.sep.join([save_model_path, 'std.th']))
            th.save(training_params, os.sep.join([save_model_path, 'training_params.th']))
            th.save(model.loss_vector, os.sep.join([save_model_path, 'loss_vector.th']))
            th.save(model.val_loss_vector, os.sep.join([save_model_path, 'val_loss_vector.th']))
            pd.DataFrame(training_params, index=[0]).T.to_csv(os.sep.join([save_model_path, 'training_params.csv']))
            # model.net.reset_parameters()  # in case multiple models with same set of parameters should be trained

        print('... Done!\n')

        # delete model instant so next model can be created
        del model
