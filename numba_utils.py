import numpy as np
from numba import njit, float64, int64

@njit 
def AddAxis(x):
    """
    Function to add axis in a numba friendly manner.
    Similar function to x[None, :]

    Parameters
    ----------
    x : 1D numpy array, floats

    Returns
    -------
    y : 2D numpy array, floats
    """
    y = np.zeros((1, x.shape[0]))
    y[0] = x
    return y

@njit
def NbRound1(x, decimals):
    """
    To replace np.round() for 1D numpy arrays

    Parameters
    ----------
    x : 1D numpy array, floats
    decimals : int
        Number of decimal places to round to (no default!). If decimals is negative, it specifies the number of positions to the left of the decimal point.

    Returns
    -------
    1D np.array with rounded values.

    """
    y = np.zeros(x.shape)
    return rnd1(x, decimals, y)
    

@njit(float64[:](float64[:], int64, float64[:]))
def rnd1(x, decimals, y):
    """
    To replace np.round() for 1D numpy arrays

    Parameters
    ----------
    x : 1D numpy array, floats
    decimals : int
        Number of decimal places to round to (no default!). If decimals is negative, it specifies the number of positions to the left of the decimal point.
     y : 1D numpy array, floats
    decimals : int
        The shape of the output array.
    
    Returns
    -------
    1D np.array with rounded values.

    """
    return np.round_(x, decimals, y)

@njit
def NbRound2(x, decimals):
    """
    To replace np.round() for 2D numpy arrays

    Parameters
    ----------
    x : 2D numpy array, floats
    decimals : int
        Number of decimal places to round to (no default!). If decimals is negative, it specifies the number of positions to the left of the decimal point.

    Returns
    -------
    2D np.array with rounded values.

    """
    y = np.zeros(x.shape)
    return rnd2(x, decimals, y)
    

@njit(float64[:, :](float64[:, :], int64, float64[:, :]))
def rnd2(x, decimals, y):
    """
    To replace np.round() for 2D numpy arrays

    Parameters
    ----------
    x : 2D numpy array, floats
    decimals : int
        Number of decimal places to round to (no default!). If decimals is negative, it specifies the number of positions to the left of the decimal point.
     y : 2D numpy array, floats
    decimals : int
        The shape of the output array.
    
    Returns
    -------
    2D np.array with rounded values.

    """
    return np.round_(x, decimals, y)

@njit(parallel = True)
def Any1(array):
    """
    Numbafyable function to replace np.any(array, axis = 1)
    
    Parameters
    ----------
    array : 2D numpy array, boolean
        shape: m x n
    Returns
    -------
    1D boolean array of size m. True where any True values in row.
    """
    size = array.shape[0]
    pre_out = np.zeros(size)
    out = pre_out > 0
    for i, row in enumerate(array):
        out[i] = row.any()
    return out

@njit(int64[:](int64[:], int64[:]))
def NotInInt(a, b):
    """
    Function to determine the integer values that are in a, 
    but not in b in a numba friendly way.

    Parameters
    ----------
    a : 1D numpy array, int
    b : 1D numpy array, int
    
    Returns
    -------
    c : 1D numpy array, int
        Containing the values that are in a, but not in b
    """
    
    return np.array([i for i in a if i not in b])

@njit(parallel = True)
def Mean0(a): 
    """
    Replacing np.mean(a, axis = 0) for 2D arrays

    Parameters
    ----------
    a : 2D numpy array, floats

    Returns
    -------
    average : 1D numpy array, floats
        Averages over the columns of a.

    """
    average = np.zeros(a.shape[1])    
    for col in range(a.shape[1]):
        average[col] = a[:,col].mean()
    return average

@njit(parallel = True)
def NanMean12(array):
    """
    Makes the sum of a nan-carrying array over the 1 and 2 axes.

    Parameters
    ----------
    array : 4D numpy array
        force, used in the BouncePointCircle function.
    axis : tuple of ints, optional
        The axes. The default is (1, 2).

    Returns
    -------
    The average, ignoring nans.
    """
    
    result = np.zeros((array.shape[0], 2)) # array[0] is the amount of balls
    for i, a in enumerate(array):
        if np.logical_not(np.isnan(a)).any(): # so no divisions by 0 occur
            counter = 0
            for p in a: 
                xy = p.T
                pre_counter = sum(np.logical_not(np.isnan(xy[0])))
                counter += pre_counter
                if not np.isnan(np.nanmean(xy[0])):
                    result[i][0] += np.nansum(xy[0])
                    result[i][1] += np.nansum(xy[1])
            result[i] /= counter
    return result 

@njit(parallel = True)
def NanMean1(array):
    """
    Makes the sum of a nan-carrying array over the 1 axis.

    Parameters
    ----------
    array : 3D numpy array
        force, used in the BouncePointCircle function.
    axis : tuple of ints, optional
        The axes. The default is (1).

    Returns
    -------
    The average, ignoring nans.
    """
    
    result = np.zeros((2))
    pre_result = np.zeros((array.shape[0], 2))
    for i, a in enumerate(array):
        for ii, arr in enumerate(a.T):
            pre_result[i][ii] = np.nanmean(arr)
    
    result = np.sum(pre_result, axis = 0)/array.shape[0]
    
    return result 
