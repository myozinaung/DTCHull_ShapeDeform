#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import supportFunctions as sF


# In[56]:


# Read the data from the csv file
# data1 = pd.read_csv('DTC_Offset_Ori.csv')
data1 = pd.read_csv('DTC_Offset_Bulb.csv')
# data2 = pd.read_csv('DTC_Offset_Scaled.csv')


# In[57]:


timeValue = np.array([0, 30, 32.5, 45, 47.5, 60])

# 4s
x1 = data1['X']
y1 = data1['Y']
z1 = data1['Z']

# delete one side(y positive) of the hull
idx = np.where(y1 > 0)
x1 = np.delete(x1, idx)
y1 = np.delete(y1, idx)
z1 = np.delete(z1, idx)

idx = np.where(z1 > 0.27)
x1 = np.delete(x1, idx)
y1 = np.delete(y1, idx)
z1 = np.delete(z1, idx)

# 0s and 4s
x0 = x1
y0 = y1
z0 = z1

# # 6s, for Hull
# x2 = x1 * 1.05
# y2 = y1 * 0.952396
# z2 = z1

# 6s, for Bulb
dMove = np.min(x1)
x1_origin = x1 - dMove # move tthe bulb to the origin
x2 = x1_origin * 1.25 # scale the bulb in x direction
x2 = x2 + dMove # move the bulb back to the original position

y2 = y1
z2 = z1

# 8s
x3 = x2
y3 = y2
z3 = z2

# 10s
x4 = x1_origin * 1.5
x4 = x4 + dMove
y4 = y3
z4 = z3

# 12s
x5 = x4
y5 = y4
z5 = z4


# In[58]:


geoPoints0 = np.stack((x0, y0, z0), axis=-1)
geoPoints1 = np.stack((x1, y1, z1), axis=-1)
geoPoints2 = np.stack((x2, y2, z2), axis=-1)
geoPoints3 = np.stack((x3, y3, z3), axis=-1)
geoPoints4 = np.stack((x4, y4, z4), axis=-1)
geoPoints5 = np.stack((x5, y5, z5), axis=-1)

# shipPoints.shape = (timePts, numPoints, 3)
shipPoints = np.stack((geoPoints0, geoPoints1, geoPoints2, geoPoints2, geoPoints4, geoPoints4), axis=0)
print(shipPoints.shape)


# In[59]:


compData = False
outputPath = "constant/boundaryData/hull"
fileName = "pointDisplacement"

for ii, tV in enumerate(timeValue):
    if ii == 0:
        outPts = shipPoints[ii, :, :]
        motDat = 0*outPts
    else:
        outPts = shipPoints[ii-1, :, :]
        motDat = (shipPoints[ii, :, :] - shipPoints[0, :, :]) # displacement from the initial position

    outDat = motDat

    # Save files
    tVal = tV

    # Save points
    print("     Exporting motion files for time t=%0.4f [s]" % tVal)

    sF.createOpenFOAMfile(data=outPts, fileName="points",
                                time=tVal, folPath=outputPath,
                                compress=compData)
    # Save data
    sF.createOpenFOAMfile(data=outDat, fileName=fileName,
                            time=tVal, folPath=outputPath,
                            compress=compData)


# In[60]:


# combine the data

fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(111, projection='3d')

# ax.view_init(elev=0, azim=0)

ax.scatter(x1, y1, z1, c='b', s=1)
ax.scatter(x2, y2, z2, c='r', s=1)
ax.scatter(x4, y4, z4, c='g', s=1)
ax.set_aspect('equal')

