import numpy as np
from numba import njit

@njit(parallel=True) # 30% faster
def Normalize(v):
    """
    Normalizes vectors so length of vector is 1.

    Parameters
    ----------
    v : 2D numpy array, floats

    Returns
    -------
    2D numpy array, floats
        Normalized v.
    """
    norm = np.zeros(v.shape[0])
    for i, vector in enumerate(v):
        norm[i] = np.linalg.norm(vector)
    norm[np.where(norm == 0)[0]] = 1  
    return v / norm.reshape(norm.size, 1) 

@njit
def CheckIntersect(L1, L2):
    """
    Checks if 2 lines intersect.

    Parameters
    ----------
    L1 : 2D numpy array, floats
        X- and y-coordinates of point representing a line.
    L2 : TYPE
        X- and y-coordinates of point representing a line..

    Returns
    -------
    boolean
        True if the lines intersect and False if not.
        
    Source
    -------
    https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
    """
    
    (X1, Y1), (X2, Y2) = L1
    (X3, Y3), (X4, Y4) = L2
    
    if max(X1, X2) < min(X3, X4) or max(X3, X4) < min(X1, X2):
        return False  
    
    if max(Y1, Y2) < min(Y3, Y4) or max(Y3, Y4) < min(Y1, Y2):
        return False # There is no mutual abcisses

    if X1 == X2: #l1 is vertical
        if X3 == X4: 
            return False
        else:
            a2 = (Y3 - Y4) / (X3 - X4) # a is slope
            b2 = Y3 - a2 * X3 # b is intersect
            Ya = a2 * X1 + b2
            if Ya < max(Y1, Y2) and Ya > min(Y1, Y2):
                return True
            else:
                return False
    if X3 == X4: #l1 is vertical
    
        a1 = (Y1 - Y2) / (X1 - X2) 
        b1 = Y1 - a1 * X1
        Ya = a1 * X3 + b1
        if Ya < max(Y3, Y4) and Ya > min(Y3, Y4):
            return True
        else:
            return False    
            
    a1 = (Y1 - Y2) / (X1 - X2)  
    a2 = (Y3 - Y4) / (X3 - X4) 
    b1 = Y1 - a1 * X1 
    b2 = Y3 - a2 * X3 
    
    if (a1 == a2):
        return False  # Parallel segments
    
    Xa = (b2 - b1) / (a1 - a2)
    
    if Xa < max(min(X1, X2), min(X3, X4)) or Xa > min(max(X1, X2), max(X3, X4)):
        return False  # intersection is outside of the segments
    else:
        return True
    
@njit
def SlopeIntercept(L):
    """
    Returns the slope and intersect with the y-axis of a line represented by two coordinates.
    Only works when L is not horiozntal or vertical.

    Parameters
    ----------
    L  : 2D np.array, floats
        A 2 x2 matrix containing 2 sets of x- and y-coordinates
        representing a line segment.

    Returns
    -------
    a : float
        Slope.
    b : float
        Intercept.
    """
    (X1, Y1), (X2, Y2) = L
    a = (Y2 - Y1) / (X2 - X1)
    b = Y1 - a * X1     
    return a, b

@njit
def OnSegment(P1, P2, P3):
    """
    Check if P3 is on line P1 - P2.

    Parameters
    ----------
    P1 : 1D numpy array, floats
        x- and y-coordinates of point. Part of the line.
    P2 : TYPE
        x- and y-coordinates of point. Part of the line.
    P3 : TYPE
        x- and y-coordinates of point. To be checked if on line.
    Returns
    -------
    bool
        True if on segment, False if not.

    Source
    -------
    https://stackoverflow.com/questions/328107/how-can-you-determine-a-point-is-between-two-other-points-on-a-line-segment/29301940#29301940
    """
    X1, Y1 = P1
    X2, Y2 = P2
    X3, Y3 = P3
    crossproduct = (Y3 - Y2) * (X2 - X1) - (X3 - X1) * (Y2 - Y1)

    if abs(crossproduct) != 0:
        return False

    dotproduct = (X3 - X1) * (X2 - X1) + (Y3 - Y1) * (Y2 - Y1)
    if dotproduct < 0:
        return False

    squaredlengthba = (X2 - X1) * (X2 - X1) + (Y2 - Y1) * (Y2 - Y1)
    if dotproduct > squaredlengthba:
        return False

    return True

@njit
def IntersectPoint(L1, L2):
    """
    Determine where the two lines L1 and L2 intersect.
    The entries are segments, but this function determines for the infinite line
    represented by that segment.

    Parameters
    ----------
    L1 : 2D numpy arra, floats
        A 2 x2 matrix containing 2 sets of x- and y-coordinates
        representing a line segment.
    L2 : 2D numpy arra, floats
        A 2 x2 matrix containing 2 sets of x- and y-coordinates
        representing a line segment.

    Returns
    -------
    Xi : float
        X-coordinate of the intersection.
    Yi : float
        Y-coordinate of the intersection.
    
    Source
    -------
    https://stackoverflow.com/questions/57044406/getting-equation-of-a-line-between-2-points-in-opencv
    """
    
    # Since this function is only used to determine where a point will intersect a polygon after being moved,
    # The second point of L1 (the point where it will be) will not incite an intersection or bounce.
    if OnSegment(L2[0], L2[1], L1[0]):
        return L1[0][0], L1[0][1]
    
    (X1, Y1), (X2, Y2) = L1
    if X1 == X2: 
        #If l1 is vertical, l2 cannot be vertical
        (X3, Y3), (X4, Y4) = L2
        a2, b2 = SlopeIntercept(L2) 
        Xi = X1
        Yi = a2 * Xi + b2
    else:
        a1, b1 = SlopeIntercept(L1)
        (X3, Y3), (X4, Y4) = L2
        if X3 == X4:
            Xi = X3
            Yi = a1 * Xi + b1
        else:
            a2, b2 = SlopeIntercept(L2)
            Xi = (b1 - b2) / (a2 - a1)
            Yi = a1 * Xi + b1
    return Xi, Yi

@njit
def MirrorPoint(PP, LP1, LP2):
    """
    Function to return the mirror point of point PP over the line between points LP1 and LP2

    Parameters
    ----------
    PP : 1D numpy array, floats
        x- and y-coordinates of point. 
    LP1 : 1D numpy array, floats
        x- and y-coordinates of point along line segment. 
    LP2 : 1D numpy array, floats
        x- and y-coordinates of point along line segment. 

    Returns
    -------
    Xm : float
        X-coordinate of the mirrorred point.
    Ym : float
        Y-coordinate of the mirrorred point.

    Source
    -------
    https://stackoverflow.com/questions/3306838/algorithm-for-reflecting-a-point-across-a-line
    """        
    X1, Y1 = PP
    X2, Y2 = LP1
    X3, Y3 = LP2
    if Y2 == Y3: #horizontal line
        Xm = X1
        Ym = Y2 - (Y1 - Y2)
    elif X3 == X2: #vertical line
        Xm = X2 - (X1 - X2)
        Ym = Y1
    else: #sloped line
        a = (Y3 - Y2) / (X3 - X2) #slope
        b = (X3 * Y2 - X2 * Y3) / (X3 - X2) #intercept
        d = (X1 + (Y1 - b) * a) / (1 + a**2)
        Xm = 2 * d - X1
        Ym = 2 * d * a - Y1 + 2 * b
    return Xm, Ym

@njit(parallel=True)
def MirrorPoints(points, intersegments):
    """
    Function to return the mirror points of the points over the line in intersegments

    Parameters
    ----------
    points : 2D numpy array, floats
        The cartesian coordinates of points.
    intersegments : 3D numpy array, floats
        For each point, the specific segment of the polygons this line is mirrored across
        is stored in this variable. Each segment is represented by two sets of 
        points (x- and y-coordinates).

    Returns
    -------
    mirrors : 2D numpy array, floats
        The locations mirorred points.

    """
    mirrors = np.zeros(points.shape)
    for i, p in enumerate(points):
        mirrors[i] = MirrorPoint(p, intersegments[i][0], intersegments[i][1])
    return mirrors

@njit
def CheckCircleIntersect(circle_center, circle_radius, P1, P2):
    """
    CHeck if a circle overlaps a certain line segment

    Parameters
    ----------
    circle_center : 1D numpy array
        The x- and y-coordinate of the circle center.
    circle_radius : float
        The radius of the circle.
    P1 : 1D numpy array, floats
        X- and y_coordinate of a point representing one of the ends of the line segement.
    P2 : 1D numpy array, floats
        X- and y_coordinate of a point representing one of the ends of the line segement.

    Returns
    -------
    bool
        True if the circle overlaps at 1 or 2 points with the line segment.

    Source
    -------
    https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    V = P2 - P1  
    a = DotProduct(V, V)
    r1 = P1 - circle_center
    b = 2 * DotProduct(V, r1)
    c = DotProduct(P1, P1) + DotProduct(circle_center, circle_center) - 2 * DotProduct(P1, circle_center) - circle_radius**2
    disc = b**2 - 4 * a * c
    if disc < 0:
        return False
    
    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    
    if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
        return False    
    return True

@njit    
def CircleLineIntersectDiscriminant(circle_center, circle_radius, P1, P2, tangent_tol = 1e-9):
    """
    Returns the values necesary to determine if a line intersects using the function full line intersections

    Parameters
    ----------
    circle_center : 1D numpy array
        The x- and y-coordinate of the circle center.
    circle_radius : float
        The radius of the circle.
    P1 : 1D numpy array, floats
        X- and y_coordinate of a point representing one of the ends of the line segement.
    P2 : 1D numpy array, floats
        X- and y_coordinate of a point representing one of the ends of the line segement.
    tangent_tol : bool, optional
        How close to the line segment is an overlap? The default is 1e-9.

    Returns
    -------
    cx : float
        X-coordinate of circle center location.
    cy : float
        Y-coordinate of circle center location.
    dx : float
        Linear distance between x-coordinates of the points in the line segment.
    dy : float
        Linear distance between y-coordinates of the points in the line segment.
    dr : float
        Length of the segment.
    big_d : float
        Parameter to determine the discriminant.
    discriminant : float
        Discriminant.

    Source
    -------
    https://stackoverflow.com/a/59582674
    https://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = P1, P2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2
    return cx, cy, dx, dy, dr, big_d, discriminant

@njit 
def Sign(dy):
    """
    Determine the sign for determining the intersection points of the circle in function FullLineIntersections().

    Parameters
    ----------
    dy : float
        Linear distance between y-coordinates of the points in the line segment.

    Returns
    -------
    1D np.array, int
        Two values corresponding to the sign value.
    """
    return np.array((1, -1) if dy < 0 else (-1, 1))

@njit
def FullLineIntersections(cx, cy, dx, dy, dr, big_d, discriminant):
    """
    Determine the intersect points of the overlap of a circle on a full line.

    Parameters
    ----------
    cx : float
        X-coordinate of circle center location.
    cy : float
        Y-coordinate of circle center location.
    dx : float
        Linear distance between x-coordinates of the points in the line segment.
    dy : float
        Linear distance between y-coordinates of the points in the line segment.
    dr : float
        Length of the segment.
    big_d : float
        Parameter to determine the discriminant.
    discriminant : float
        Discriminant.

    Returns
    -------
    intersections : 2D numpy array
        A 2x2 numpy array containing two sets of locations (x, y) of the intersect. 
        If only one intersect is present, both have the same value.

    """
    intersections = np.zeros((2,2))
    for i, sign in enumerate(Sign(dy)):
        intersections[i][0] = cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2
        intersections[i][1] = cy + (-big_d * dx + sign * abs(dy) * np.sqrt(discriminant)) / dr ** 2
    return intersections
