#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# %% Header
# =============================================================================
''' Create a geometry and trailing edge deformation for a NACA 4-digit airfoil.

Created By  : André Da Luz Moreira, Linköping University/Sweden
Contact     : andre.da.luz.moreira@liu.se
Created Date: 16/Dec/2022
Version     : 0.0.1

These are support functions used to create geometry and motion files for
tutorials using case-sepecific python scripts.

Created as part of the course CFD with Open Source Software offered by
Chalmers University of Technology in 2022.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.

This is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.  See the file LICENSE in this directory or
[http://www.gnu.org/licenses/](http://www.gnu.org/licenses), for a
description of the GNU General Public License terms under which you
may redistribute files.
'''

__author__ =        "André Da Luz Moreira"
__contact__ =       "andda15@liu.se"
__copyright__ =     "Copyright 2022, André Da Luz Moreira"
__credits__ =       ["André Da Luz Moreira"]
__date__ =          "2022/12/16"
__deprecated__ =    False
__email__ =         "andre.da.luz.moreira@liu.se"
__repository__ =    "https://gitlab.liu.se/andda15/timevaryingmotioninterpolation"
__license__ =       "GPLv3"
__status__ =        "Development"
__version__ =       "0.0.1"

# =============================================================================
# %% Import dependencies
# =============================================================================
import os
import gzip
import numpy as np
import scipy.optimize as optimize

# =============================================================================
# %% Functions
# =============================================================================


def makeCoordinates(xx, yy, zz=None, transpose=True):
    '''
    Creates a matrix of shape (N,3) containing the point coordinates given as
    inputs.

    Inputs:
        xx, yy, zz
            Vectors or matrices of N elements with the X,Y,Z coordinates.
        transpose [OPTIONAL]
            If true, the output is a matrix of dimensions Nx3.  If false, the
            output is 3xN.
    Outputs:
        coordinates
            A matrix containing the combined point coordinates.
    '''
    if zz is None:
        coordinates = np.array([xx.flatten(), yy.flatten(), 0*yy.flatten()])
    else:
        coordinates = np.array([xx.flatten(), yy.flatten(), zz.flatten()])

    if transpose:
        return coordinates.T
    else:
        return coordinates

# =============================================================================


def createOpenFOAMfile(data, fileName="OFfile", time=-1, folPath=".", compress=False):
    """
    Export data to an OpenFOAM readable file.

    Inputs:
        data        -> A (N,3) numpy array containing the data to be exported.
        fileName    -> Output file name.
        time        -> Data time value (if not provided, data in folPath).
        folPath     -> Folder where data will be saved.
        compress    -> If true, data is saved as .gz.
    """
    # Create/check necessary folder(s)
    os.makedirs(folPath, exist_ok=True)
    if time > -1:
        fileFolder = os.path.join(folPath, "%0.4f" % time)
        os.makedirs(fileFolder, exist_ok=True)
    else:
        fileFolder = folPath
    # Save file
    filePath = os.path.join(fileFolder, fileName)
    if compress:
        dataOutput = gzip.open(filePath+".gz", 'wt')
    else:
        dataOutput = open(filePath, "w+")
    dataOutput.write("// %s data at t=%0.4f [s].\n" % (fileName, time))
    dataOutput.write("%d\n(\n" % data.shape[0])
    dataOutput.writelines(['(%0.8f %0.8f %0.8f)\n' % tuple(vv) for vv in data])
    dataOutput.write(")\n")
    dataOutput.close()

# =============================================================================


def createCoordinatesDataFile(data, time=-1, folPath=".", fileName="points", compress=False):
    """
    Export data to an OpenFOAM readable file.

    Inputs:
        data        -> A (N,3) numpy array containing the data to be exported.
        time        -> Data time value (if not provided, data in folPath).
        folPath     -> Folder where data will be saved.
        fileName    -> Output file name.
        compress    -> If true, data is saved as .gz.
    """
    # Create/check necessary folder(s)
    os.makedirs(folPath, exist_ok=True)
    if time > -1:
        fileFolder = os.path.join(folPath, "%0.4f" % time)
        os.makedirs(fileFolder, exist_ok=True)
    else:
        fileFolder = folPath
    # Save file
    filePath = os.path.join(fileFolder, fileName)
    if compress:
        dataOutput = gzip.open(filePath+".gz", 'wt')
    else:
        dataOutput = open(filePath, "w+")
    dataOutput.write("// Displacement points at t=%0.4f [s].\n" % (time))
    dataOutput.write("%d\n(\n" % data.shape[0])
    dataOutput.writelines(['(%0.8f %0.8f %0.8f)\n' % tuple(vv) for vv in data])
    dataOutput.write(")\n")
    dataOutput.close()

# =============================================================================


def createMotionDataFile(data, time, folPath=".", fileName="pointMotionU", compress=False):
    # Create/check necessary folder(s)
    os.makedirs(folPath, exist_ok=True)
    fileFolder = os.path.join(folPath, "%0.4f" % time)
    os.makedirs(fileFolder, exist_ok=True)
    # Output data to file
    filePath = os.path.join(fileFolder, fileName)
    if compress:
        dataOutput = gzip.open(filePath+".gz", 'wt')
    else:
        dataOutput = open(filePath, "w+")
    dataOutput.write(
        "// Displacement field %s at t=%0.4f [s].\n" % (fileName, time))
    dataOutput.write("%d\n(\n" % data.shape[0])
    dataOutput.writelines(['(%0.8f %0.8f %0.8f)\n' % tuple(vv) for vv in data])
    dataOutput.write(")\n")
    dataOutput.close()

# =============================================================================


def createDomainMatrixInfo(matPoint, matSpace, matShape, folPath="."):
    # Create/check necessary folder(s)
    os.makedirs(folPath, exist_ok=True)
    # Output domainMatrixInfo file
    filePath = os.path.join(folPath, 'domainMatrixInfo')
    with open(filePath, 'w+') as dataOutput:
        dataOutput.write("// Motion data matrix information.\n")
        dataOutput.write("// Line 1: reference point (x,y,z).\n")
        dataOutput.write("// Line 2: voxels sizing (dx,dy,dz).\n")
        dataOutput.write("// Line 3: matrix dimensions (nx,ny,nz).\n")
        dataOutput.write("3\n(\n")
        dataOutput.write('(%0.8f %0.8f %0.8f)\n' %
                         (matPoint[0], matPoint[1], matPoint[2]))
        dataOutput.write('(%0.8f %0.8f %0.8f)\n' %
                         (matSpace[0], matSpace[1], matSpace[2]))
        dataOutput.write('(%d %d %d)\n' %
                         (matShape[0], matShape[1], matShape[2]))
        dataOutput.write(")\n")

# =============================================================================


def nacaSym(tt, disc=100, xCos=True, closeTe=False):
    """
    Calculates coordinates for a symmetric NACA 4-digit airfoil.

    Airfoil code is 00tt, where tt is the thickness in percentage of chord.
    For details, refer to chapter 6 in

        I. H. Abbott and A. E. V. Doenho , Theory of Wing Sections, Including
        a Summary of Airfoil Data. Dover Publications, Jan. 1959.

    Inputs:
        tt      -> thickness in percentage of chord
        cc      -> chord length (scales xx values)
        disc    -> discretisation in xxx
        xCos    -> use a cosine term for the X-coordinates?
        closeTe -> closed trailing edge?
    Outputs:
        xx      -> array of X-coordinates
        y_t     -> array of Y-coordinates for upper surface
    """
    # Convert input to length
    tt /= 100
    # Define the X-Coordinates
    if xCos:
        beta = np.linspace(0, np.pi, disc)
        xx = 0.5*(1-np.cos(beta))
    else:
        xx = np.linspace(0, 1, disc)
    # Naca airfoil shape constants (Abbot, Eq. 6.2)
    a_0 = 0.2969
    a_1 = -0.1260
    a_2 = -0.3516
    a_3 = 0.2843
    if closeTe:
        a_4 = -0.1036
    else:
        a_4 = -0.1015
    # Upper surface Y-Coordinates (Abbot, Eq. 6.2)
    y_t = 5*tt*(a_0*np.sqrt(xx)+a_1*xx+a_2*xx**2+a_3*xx**3+a_4*xx**4)

    return xx, y_t

# =============================================================================


def naca4dig(mm, pp, tt, yTE=0, teLen=None, disc=100, xCos=True, closeTe=False, flipTopx=True):
    """
    Calculates coordinates for a NACA 4-digit airfoil.

    For details, refer to chapter 6 in

        I. H. Abbott and A. E. V. Doenho , Theory of Wing Sections, Including
        a Summary of Airfoil Data. Dover Publications, Jan. 1959.

    In case the trailing edge length is provided, the trailing section of the
    airfoil is scaled to maintain its size.

    Inputs:
        mm      -> maximum camber (percentage of chord)
        pp      -> position of maximum camber (tenths of chord)
        tt      -> thickness (percentage of chord)
        yTE     -> the y-coordinate of the trailing edge point (default 0)
        teLen   -> the length of the trailing section of the airfoil
        disc    -> discretisation in xx (for upper and lower)
        xCos    -> use a cosine term for the X-coordinates?
        closeTe -> closed trailing edge?
        flipTopx-> flip the arrays for upper coordinates?
                   (sequential points starting and finishing at TE)
    Outputs:
        x_U     -> array of X-coordinates for upper surface
        y_U     -> array of Y-coordinates for upper surface
        x_L     -> array of X-coordinates for lower surface
        y_L     -> array of Y-coordinates for lower surface
    """
    # Get the X and Y-coordinates for the symmetrical NACA airfoil
    xx, y_t = nacaSym(tt=tt, disc=disc, xCos=xCos, closeTe=closeTe)
    # Check if it's a symmetric airfoil
    if mm == 0 and pp == 0:
        x_U = xx
        x_L = xx
        y_U = y_t
        y_L = -y_t
        # Return upper surface from TE to LE?
        if flipTopx:
            x_U = np.flip(x_U)
            y_U = np.flip(y_U)
        return x_U, y_U, x_L, y_L
    # Indices for the two airfoil sections
    indicesA = np.where(xx <= 0.1*pp)
    indicesB = np.where(xx >= 0.1*pp)
    # Scale the trailing edge to maintain camber length (ONLY TE!)
    if teLen is not None:
        fInputs = (xx[indicesB], mm, pp, teLen, 1, yTE)
        xGuess = (1)
        bnds = ((0, 2),)
        sol = optimize.minimize(teScaling,
                                x0=xGuess,
                                bounds=bnds,
                                method="Nelder-Mead",
                                args=fInputs,
                                tol=1e-8)
        xx[indicesB] = 0.1*pp + sol.x*(xx[indicesB]-0.1*pp)
    # Chord coordinates
    y_c = np.empty(shape=xx.shape)
    y_c[indicesA] = naca4dig_camber(xx[indicesA], mm, pp, x2=0, y2=0)   # LE
    y_c[indicesB] = naca4dig_camber(xx[indicesB], mm, pp, x2=1, y2=yTE) # TE
    # Derivative of chord with respect to X
    dYc_dx = np.empty(shape=xx.shape)
    dYc_dx[indicesA] = naca4dig_camberDer(
        xx[indicesA], mm, pp, x2=0, y2=0)       # LE
    dYc_dx[indicesB] = naca4dig_camberDer(
        xx[indicesB], mm, pp, x2=1, y2=yTE)     # TE
    # Angles for tangents
    theta = np.arctan(dYc_dx)
    # Upper and lower surfaces (Abbot, Eq. 6.1)
    x_U = xx - y_t*np.sin(theta)
    x_L = xx + y_t*np.sin(theta)
    y_U = y_c + y_t*np.cos(theta)
    y_L = y_c - y_t*np.cos(theta)
    # Return upper surface from TE to LE?
    if flipTopx:
        x_U = np.flip(x_U)
        y_U = np.flip(y_U)

    return x_U, y_U, x_L, y_L

# =============================================================================


def naca4dig_camber(xx, mm, pp, x2=1, y2=0):
    """
    Returns camber y_c coordinate(s) corresponding to the x-coordinate(s) xx.

    The NACA 4-digit camber is calculated as two parabolae meeting at the
    maximum camber point, which is the vertex in both, maintaining the curves'
    continuity.

    This funciton uses the parabolic equation in vertex form,
        y = a * (x-V_x)^2 + V_y ,
    for a parabola with a vertex point (V_x,V_y).

    The constant a is calculated from another point in the curve. For NACA
    4-digit airfoils this point is is (0,0) or (1,0) for the leading and
    trailing edges, respectively.

    Inputs:
        xx      -> X-coordinate(s)
        mm      -> maximum camber (percentage of chord)
        pp      -> position of maximum camber (tenths of chord)
        x2,y2   -> coordinates of second point in parabola fit [default (1,0)]
    Output:
        y_c     -> the camber y-coordinate(s) corresponding to inputs
    """
    # Parabola's vertex
    V_x = 0.1*pp
    V_y = 0.01*mm
    # Constant a
    a = (y2-V_y)/((x2-V_x)**2)
    # Curve
    return a*((xx-V_x)**2)+V_y

# =============================================================================


def naca4dig_camberDer(xx, mm, pp, x2=1, y2=0):
    """
    Returns the derivative of the camber curve corresponding to the
    x-coordinate(s) xx.

    The derivative of the parabolic equation in vertex form is
        dy_dx = 2*a * (x-V_x) ,
    for a parabola with a vertex point (V_x,V_y).

    The constant a is calculated from another point in the curve. For NACA
    4-digit airfoils this point is is (0,0) or (1,0) for the leading and
    trailing edges, respectively.

    Inputs:
        xx      -> X-coordinate(s)
        mm      -> maximum camber (percentage of chord)
        pp      -> position of maximum camber (tenths of chord)
        x2,y2   -> coordinates of second point in parabola fit [default (1,0)]
    Output:
        dy_dx     -> the camber y-coordinate(s) corresponding to inputs
    """
    # Parabola's vertex
    V_x = 0.1*pp
    V_y = 0.01*mm
    # Constants
    a = (y2-V_y)/((x2-V_x)**2)
    # Derivative
    return 2*a*(xx-V_x)

# =============================================================================


def teScaling(fPars, *fInputs):
    """
    Function used for scaling of trailing section of NACA 4-digit aifoil by
    matching a specific length of the camber line.

    See function naca4dig_camber for details.

    Parameter:
        sF      -> scaling factor applied to x-coordinates of trailing edge

    Inputs:
        xx      -> X-coordinates of camber points
        mm      -> maximum camber (percentage of chord)
        pp      -> position of maximum camber (tenths of chord)
        teLen   -> trailing edge length
        xTE,yTE -> x and y coordinates used in the camber parabola

    """
    # Function minimisation parameter:
    sF = fPars
    # Inputs
    xx, mm, pp, teLen, xTE, yTE = fInputs
    # Correct x-coordinates
    xCor = 0.1*pp + sF*(xx-0.1*pp)
    # Calculate trailing camber points
    yCam = naca4dig_camber(xCor, mm, pp, x2=xTE, y2=yTE)
    # Length of camber curve from vertex point to TE
    distPts = np.sqrt((xCor[1:]-xCor[:-1])**2+(yCam[1:]-yCam[:-1])**2)
    # Return the difference between calculated distance and provided value
    return abs(np.sum(distPts)-teLen)

# =============================================================================
# %% End
# =============================================================================
