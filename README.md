# data_driven_rans:

Repository on Master's thesis "Machine Learning Augmented Turbulence Modelling for Reynolds Stress Clorsure Problem"

## Introduction

This repository contains a pytorch based implementation of the Tensor Basis Neural Network proposed by Ling et al. [[1]](#1). While the network architeckture is the same, the feateure set was extended as in Wu et al. [[2]](#2), [[3]](#3).

What the code in this repository can do:
- Read in RANS data from OpenFOAM, extract scalar invariants, and compute tensor basis
- Read in preprocessed DNS/LES data stored in .th files and interpolate them onto RANS grid
- Store both RANS and DNS/LES data so it can be accessed for training
- Select data and train the TBNN to find a mapping from invariant input features from RANS to labels (anisotropy tensor) from DNS/LES
- Store trained model so it can be accessed for prediction
- Create OpenFOAM file for the anisotropy tensor to be used as source term for baseline RANS equation
- Compute barycentric map coordinates [[4]](#4) and visualize anisotropy tensor predictions

## Dependencies

The following python packages are used and can be installed by executing the following commands:

```
pip3 install numpy pandas matplotlib scikit-learn torch torchvision scipy seaborn
```

## References
<a id="1">[1]</a> 
J. Ling, A. Kurzawaski, J. Templetom (2016).  
Reynolds averaged turbulence modelling using deep neural networks with embedded invariance,
Journal of Fluid Mechanics, 807, 155-166.

<a id="2">[2]</a> 
J. Wu, H. Xiao, E. Paterson (2018).  
Physics-informed machine learning approach for augmenting turbulence models: A comprehensive framework,
Physical Review Fluids, 7(3), 74602.

<a id="3">[3]</a> 
J. Wang, J. Wu, J. Ling, G. Iaccarino, H. Xiao (2017).  
A comprehensive physics-informed machine learning framework for predictive turbulence modeling,
arXiv.

<a id="4">[4]</a> 
S. Banjaree, R. Krahl, F. Durst, CH. Zenger (2007).  
Presentation of anisotropy properties of turbulence, invariants versus eigenvalue approaches,
Journal of Turbulence, 8, 1-27
