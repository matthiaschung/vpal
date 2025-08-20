# Variable Projection Augmented Lagrangian
This repository includes code for the variable projected augmented lagrangian algorithm (vpal) suitable for solving inverse problems. Implementations using the gradient step (vpal.m) and the non-linear conjugate gradient step (vpalnl.m) are included.  For further details on the algorithms, see https://www.sciencedirect.com/science/article/abs/pii/S0168927423001800 and https://iopscience.iop.org/article/10.1088/1361-6420/addde4/meta, respectively.

## Authors
Matthias Chung, Rosemary Renaut, Jack Michael Solomon

## Requirements
-MATLAB (R2024a or newer)

## Usage
To setup the file structure, begin by running the script
```matlab pathsetup.m

A denoising example can be found in `demo/` which demonstrates the use of both vpal.m and vpalnl.m.

Both implementations are reliant on custom operators included in the dOperator.m file.  See dOperator.m for further details.

## EEG Source Localization Demo
To be implemented

## License
License details can be found in LICENSE
