import numpy as np
from numba import float64, int64
from numba.experimental import jitclass
from linalg_utils import CheckIntersect, IntersectPoint

def make_numba_array(poly):
    """
    Numba and numpy do not do ragged nested sequences (for instace, a list of list with different lengths).
    This function converts ragged nested lists of cartesian coordenates into numba usable 3D numpy array.
    It achieves this by lengthening the shorter values to the longest one, and filling it with np.NaN values.

    Parameters
    ----------
    poly : 3D list, floats
        A list of list of list with two valyes of any length, 
        representing a series of polygons, of which the first one is the boundary around them all 
        and the rest are holes in it.

    Returns
    -------
    shapes : 3D numpy array, floats
        numba readably polygon.

    """
    shape_points = np.zeros(len(poly))
    for i, p in enumerate(poly):
        shape_points[i] = int(p.shape[0])
    shapes_shape = (len(poly), int(np.max(shape_points)), 2)
    shapes = np.zeros(shapes_shape)
    
    for i in range(len(shapes)):
        for ii in range(len(shapes[i])):
            if ii < len(poly[i]):
                shapes[i][ii] = poly[i][ii]
            else:
                shapes[i][ii] = (np.nan, np.nan)
    return shapes

# Determine the variable types entering this numba.jitclass
poly_spec = [('polygons', float64[:, :, :]),
             ('n', int64),
             ('entries', int64),
             ('x', float64[:, :]),
             ('y', float64[:, :]),
             ('centringpoint', float64[:])]

@jitclass(poly_spec)
class Polygon(object):
    """
    Attributes
    ----------
    polygons : 3D numpy array, floats
        A 3D numpy array containing the cartesian coordinates of all the polygons present.
        The first polygon is the boundary around them all.
        Polygons must be closed (last point is the same as the first point)
    n  : int
        The number of polygons in polygons.
    entries : int
        The length of the points of the polygons. If polygons is a n x m x 2 matrix, entries is m.
    x  : 2D numpy array, floats
        All the x-values, of each polygon in polygons
    y  : 2D numpy array, floats
        All the y-values, of each polygon in polygons
    centringpoint : 1D numpy array, floats
        X- and y-coordinates of the midpoint of the polygons, or when this falls outside the polygon,
        A random point in it.
    """    
    def __init__(self, polygons):
        """
        Initializes the polygon class object.

        Parameters
        ----------
        polygons : 3D numpy array, floats
            A 3D numpy array containing the cartesian coordinates of all the polygons present.
            The first polygon is the boundary around them all.
        """
        self.polygons = polygons
        self.n = polygons.shape[0]
        self.entries = polygons.shape[1]
        self.x = self.get_x()
        self.y = self.get_y()
        self.centringpoint = self.CentringPoint()
    
    def get_x(self):
        """
        Get all the x-values of the polygons.

        Returns
        -------
        x_ : 2D numpy array, floats
            A self.n by self.entries matrix with all x-values.
        """
        x_ = np.zeros((self.n, self.entries))
        for i, pol in enumerate(self.polygons):
            x_[i] = pol[:, 0]
        return x_
    
    def get_y(self):
        """
        Get all the y-values of the polygons.

        Returns
        -------
        y_ : 2D numpy array, floats
            A self.n by self.entries matrix with all y-values.
        """
        y_ = np.zeros((self.n, self.entries))
        for i, pol in enumerate(self.polygons):
            y_[i] = pol[:, 1]
        return y_
    
    def CentringPoint(self):
        """
        Returns
        -------
        midpoint : 1D numpy array, floats
            The x- and y-coordinates of the midpoint of all polygons. If this 
            midpoint does not fall within the polygon, return random point in it.
        """
        x = np.nanmin(self.x) + (np.nanmax(self.x) - np.nanmin(self.x)) / 2
        y = np.nanmin(self.y) + (np.nanmax(self.y) - np.nanmin(self.y)) / 2
        midpoint = np.array((x, y))
        midpoint_ = np.zeros((1, 2))
        midpoint_[0] = midpoint
        while not self.ContainsPoints(midpoint_):
            x = np.random.uniform(np.nanmin(self.x), np.nanmax(self.x))
            y = np.random.uniform(np.nanmin(self.y), np.nanmax(self.y))
            midpoint = np.array((x, y))
            midpoint_[0] = midpoint
        return midpoint
        
    def RayTracing(self, P, polygon):
        """
        Checks if a point falls within a polygon areas.

        Parameters
        ----------
        P : 1D numpy array, floats
            X- and y-coordinates of a point.
        polygon : 2D numpy array.
            A self.n x 2 matrix with the x-and y-coordinates of the 
            polygon.

        Returns
        -------
        inside : boolean
            True if in the polygon, False if not.
            
        Source
        -------
        https://stackoverflow.com/a/48760556

        """
        X, Y = P
        n = len(polygon)
        inside = False
    
        PX1, PY1 = polygon[0]
        for i in range(n+1):
            PX2, PY2 = polygon[i % n]
            if Y > min(PY1, PY2) and Y <= max(PY1, PY2) and X <= max(PX1, PX2):
                if PY1 != PY2:
                    xints = (Y - PY1) * (PX2 - PX1) / (PY2 - PY1) + PX1
                if PX1 == PX2 or X <= xints:
                    inside = not inside
            PX1, PY1 = PX2, PY2
        return inside

    def ContainsPoints(self, points):
        """
        Checks if points falls within the self.polygons area.
        
        If it falls within the first polygon, but not in the tothers, this
        function returns True.
        If it falls within the first polygon, but also in any others, it
        returns falls.
        If it falls outside any polygons, return False.

        Parameters
        ----------
        points : 2D numpy array, floats
            X- and y-coordinates of points.

        Returns
        -------
        inside : 1D np.array, boolean
            True if in the polygon, False if not.
        """
        pre_isin = np.zeros((self.n, len(points)))
        isin = pre_isin > 0
        for i, polygon in enumerate(self.polygons):
            isin[i] = np.array([self.RayTracing(point, polygon) for point in points])
            
        is_in_check = isin.T
        pre_check = np.zeros(len(points))
        check = pre_check > 0
        for i, c in enumerate(is_in_check):
            check[i] = c[0] and not c[1:].any()      
        return check
    
    def PolygonIntersection(self, lines):
        """
        Checks if a line intersects any of the sides in the self.polygons polygons.

        Parameters
        ----------
        lines : 3D numpy array, floats
            a list of lines,  where two sets of x- and y-coordinates represent a line.

        Returns
        -------
        intersect : 2D numpy array, floats
            A len(lines) by 2 matrix containing the x- and y-coordinates of the 
            point where the line and polygon intersect.
        intersegments : 3D numpy array, floats
            For each line, the specific segment of the polygons this line intersects with
            is stored in this variable. Each segment is represented by two sets of 
            points (x- and y-coordinates).

        """
        intersegments = np.zeros((lines.shape[0], 2, 2))
        intersect = np.zeros((lines.shape[0], 2))
        for i, point in enumerate(lines):
            for polygon in self.polygons:
                for seg in range(polygon.shape[0]-1):
                    segment = polygon[seg:seg+2]
                    if CheckIntersect(point, segment) and not np.isnan(segment).any():     
                        xi, yi = IntersectPoint(point, segment)
                        intersect[i] = xi, yi 
                        intersegments[i] = segment               
        return intersect, intersegments