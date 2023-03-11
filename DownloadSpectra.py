#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 18:17:32 2023

@author: gabriel
"""

import pandas as pd
import numpy as np

Catalog = pd.read_csv("CatalogPMJ.csv", sep=";")

Plate = np.ones(Catalog["PMJ"].size, dtype=object)
MJD = np.ones(Catalog["PMJ"].size, dtype=object)
Fiber = np.ones(Catalog["PMJ"].size, dtype=object)
url = np.ones(Catalog["PMJ"].size, dtype=object)

for i in range(Catalog["PMJ"].size):
    Plate[i] = Catalog["PMJ"][i][0:4]
    MJD[i] = Catalog["PMJ"][i][5:10]
    Fiber[i] = Catalog["PMJ"][i][11:15]
    if Catalog["PMJ"][i][11:15][0] == 0:
        Fiber[i] = Fiber[i][1:]
    url[i] = "http://dr16.sdss.org/optical/spectrum/view/data/format=csv?plateid=" + \
        Plate[i]+"&mjd="+MJD[i]+"&fiberid="+Fiber[i]+"&reduction2d=v5_7_0"

RetryList = []

for i in range(Catalog["PMJ"].size):
    arqv = Catalog["PMJ"][i]
    tabela = pd.read_csv(url[i])
    try:
        tabela.to_csv("./SDSS/"+arqv+".csv", index=False)
    except:
        RetryList.append(arqv)
for i in range(len(RetryList)):
    arqv = RetryList[i]
    tabela = pd.read_csv(url[i])
    try:
        tabela.to_csv("./SDSS/"+arqv+".csv", index=False)
    except:
        RetryList.append(arqv)