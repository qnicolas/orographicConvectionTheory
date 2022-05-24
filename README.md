This folder contains data and code to solve for equations and reproduce figures appearing in Nicolas, Q., & Boos, W. R. (2022). A theory for the response of tropical moist convection to mechanical orographic forcing, Journal of the Atmospheric Sciences. 

The main file for reproducing figures is makeFigures.ipynb. It uses functions from orographicConvectionTheory.py to solve for the various precipitation equations.
Data is stored in netCDF format; WRF variables use native names unless there is no corresponding native variable (e.g. 'precip', 'radiativecooling'). Data have been interpolated to pressure levels and averaged in Time and in the south_north dimension. 

For any questions, contact qnicolas@berkeley.edu 
