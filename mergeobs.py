#file to merge time dimensions of observations

import numpy as np
import cdo 
cdo = cdo.Cdo()

path = '/work/uo1075/pool/data/EN4/Resort/'

for j in np.arange(1958, 2021):
    cdo.mergetime(['en4' + str(j) + str(i) + '_1744x872_GR15L40.nc' for i in range(12)], output='en4' + str(j) + '1_12_1744x872_GR15L40.nc')