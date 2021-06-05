#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
from netCDF4 import Dataset
import keras as ks

path = '/home/jinma/dms_data/satellite'
file1 = 'A2002194.L3b_DAY_RRS.nc'
file2 = 'A2002196.L3m_DAY_RRS_angstrom_4km.nc'
file3 = 'A2003193.L3m_DAY_CHL_chlor_a_4km.nc'
nc1 = Dataset(os.path.join(path, file1), 'r')
nc2 = Dataset(os.path.join(path, file2), 'r')
nc3 = Dataset(os.path.join(path, file3), 'r')

print nc1
print nc2
print nc3.groups

print "new job is done."
print 'new test.'
print 'qweddd'