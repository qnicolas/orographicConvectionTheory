This folder contains data and code to solve for equations and reproduce figures appearing in Nicolas, Q., & Boos, W. R. (2022). A theory for the response of tropical moist convection to mechanical orographic forcing, Journal of the Atmospheric Sciences. 

The main file for reproducing figures is makeFigures.ipynb. It uses functions from orographicConvectionTheory.py to solve for the various precipitation equations.
Data is stored in netCDF format; WRF variables use native names unless there is no corresponding native variable (e.g. 'precip', 'radiativecooling'). Data have been interpolated to pressure levels and averaged in Time and in the south_north dimension. 

A .yml file is included that contains all necessary python packages to run the code and produce the figures. Create a conda environment using `conda env create -f environment.yml`, then activate with `conda activate orogconv`, launch a Jupyter notebook and you are hopefully all set!

Initial release : [![DOI](https://zenodo.org/badge/495957541.svg)](https://zenodo.org/badge/latestdoi/495957541)

For any questions, contact qnicolas@berkeley.edu 
