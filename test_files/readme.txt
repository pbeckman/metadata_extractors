2014-August-14

*** Updated README file ***

========================================================================
Temporal Improvements for Modeling Emissions by Scaling (TIMES)
and Canadian province spatial redistribution scale factors
========================================================================

These files contain temporal scale factors for diurnal and weekly modes 
of variability and scale factors to improve the spatial distribution
of Canadian fossil fuel emissions at the provincial scale.

------------------------------------------------------------------------
a) Temporal scale factors for diurnal and weekly modes of variability

These scale factors can be applied to gridded emission data products 
from CDIAC, ODIAC, EDGAR or others which lack diurnal and weekly
variability.  Scale factors with values > 1 increase emissions at times
of the day/week when emissions are higher and scale factors < 1 
decrease emissions at times of the day/week when emissions are lower.

A regular rectangular grid is used with 0.25 deg x 0.25 deg resolution. 
The first value of the 2-D array is located at 89.875 S and 179.875 W.

For the diurnal files, there are 24 values for hrs 1-24 Universal Time.

For the weekly file, days are ordered as:
1 - Monday
2 - Tuesday
3 - Wednesday
4 - Thursday
5 - Friday
6 - Saturday
7 - Sunday

Files are global (90 S - 90 N and 180 W - 180 E).
Over oceans and other major bodies of water scale factors are always 1.

------------------------------------------------------------------------
b) Spatial scale factors for Canada

The spatial scale factors adjust ODIAC or CDIAC to agree with the 
provincial weighting from Canada's National Inventory Report (2011).

A regular rectangular grid is used with 1.0 deg x 1.0 deg resolution. 
The first value of the 2-D array is located at 89.5 S and 179.5 W.

For all other regions of the world, scale factors are always 1.

------------------------------------------------------------------------

Additional information is found in the original README file below.
Detailed information about the method is given in the reference below.
Mapping the data and comparing to Figures 3 and 4 will confirm that the
grids were interpreted correctly.


*** Original README file ***

2012-November-28

------------------------------------------------------------------------
Temporal Improvements for Modeling Emissions by Scaling (TIMES)
------------------------------------------------------------------------
diurnal_scale_factors.nc
diurnal_std_dev.nc
weekly_factors_scale_China.nc
weekly_std_dev_scale_China.nc
weekly_factors_constant_China.nc
weekly_std_dev_constant_China.nc

NOTES:

The std_dev files give the standard deviation of the scaling factors
which can be used as a measure of the uncertainty.

Due to the lack of a weekly cycle in NO2 over China based on satellite 
observations, a version of weekly scale factors with no diurnal
variability over China and its proxy countries is also available.

------------------------------------------------------------------------
Canadian province spatial redistribution scale factors based 
on Canada's National Inventory Report (NIR) for 2011
------------------------------------------------------------------------
Canada_NIR_CDIAC_scale_factors.nc
Canada_NIR_ODIAC_scale_factors.nc

------------------------------------------------------------------------
Please see the following paper for more details:

Nassar, R., L. Napier-Linton, K.R.R. Gurney, R. Andres, T. Oda, F.R. 
Vogel, and F. Deng, Improving the temporal and spatial distribution of 
CO2 emissions from global fossil fuel emission datasets, J. Geophys. 
Res., 118, 917-933, 2013, doi:10.1029/2012JD018196.

------------------------------------------------------------------------

If you use these files in your research leading to publication, please
contact me to determine if co-authorship or citation is appropriate.
Additionally, please contact me if you find significant errors.

Ray Nassar
Climate Research Division, Environment Canada
ray.nassar@ec.gc.ca

