## Short doc of neural network trial:

**Part 1: Reading in Data**

In the first couple of cells (2-5), fluid data is read in. Data-selection happened in `select_nn_data.ipynb`.
Data is a subset of the fluid domain and covers the interval x in [3, 4], y in [1, 2]. Domain thus is a square.

I also defined a sigmoid function that maps tensor values to the inteval [-1, 1]. Tensor T is normalized by its spectral norm.

**Part 2: custom NN class for TBNN**

Neural network structure and forward propagation is defined first.

After that, a forward model is defined.

Next, model instance is created and the data is split into training and test data.

L2-Loss function for the weights to introduce regularization is defined. Currently not used.

Definition of training parameters such as lerning rate, optimizer, batchsize and epochs are defined.

Optimization loop:
1. loop over epochs. training data is randomly permuted before every epoch.
2. loop over batches until epoch is finished.

**Part 3: visualization of inputs and outputs**

1. Visualizing scaled invariants. 5 plots in total for all five invariants.

2. Visualizing tensor basis T. There is 10 figures (number of basis tensors) with 9 plots (each tensor component).

3. Visualizing anisotropy reynolds-stress (output of NN). There is 9x2 plots.
This is for the 9 tensor components. For every component, b_pred and b_dns is shown.

**Part 4: implementation of a more generic NN to change structure more easily**

Currently not used.