# -*- coding: utf-8 -*-
import os
import pandas as pd


data_dir = '/home/ilya/code/mllc/data/dataset_OGLE/indexes_normalized'
file_1 = 'vast_lightcurve_statistics_normalized_variables_only.log'
file_0 = 'vast_lightcurve_statistics_normalized_constant_only.log'
names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']
df_1 = pd.read_table(os.path.join(data_dir, file_1), names=names,
                     engine='python', na_values='+inf', sep=r"\s*",
                     usecols=range(30))
df_0 = pd.read_table(os.path.join(data_dir, file_0), names=names,
                     engine='python', na_values='+inf', sep=r"\s*",
                     usecols=range(30))

