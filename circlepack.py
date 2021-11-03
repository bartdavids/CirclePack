import numpy as np
from numba import float64, int64, boolean, deferred_type
from numba.experimental import jitclass
from polygon import Polygon
from numba_utils import NanMean12, NbRound1, NbRound2, Mean0, Any1, NotInInt
from linalg_utils import CheckCircleIntersect, Normalize, CircleLineIntersectDiscriminant, FullLineIntersections, MirrorPoints 

poly_type = deferred_type()
poly_type.define(Polygon.class_type.instance_type)

pack_spec = [('x', float64[:]),
             ('y', float64[:]),
             ('r', float64[:]),
             ('n', int64),
             ('precision', int64),
             ('on_point', boolean),
             ('polygons', poly_type),
             ('dx', float64[:, :]),
             ('dy', float64[:, :]),
             ('d', float64[:, :]),
             ('dr', boolean[:, :]),
             ('vx', float64[:]),
             ('vy', float64[:])]

@jitclass(pack_spec)
class CirclePack(object):
    """
    Class storing information on circle properties
    
    Attributes
    ----------
    x : 1D numpy array, floats
        The x-coordinates of the circle centres.
        Necesary to create class object, inititated on class creation. 
    y : 1D numpy array, floats
        The y-coordinates of the circle centres.
        Necesary to create class object, inititated on class creation.
    r : 1D numpy array, floats
        Necesary to create class object, inititated on class creation.
    precision: int
        The accuracy to which is rounded. 
        This is important for determining wether a point either intersects with a line, or is just very close to it.
        Default is 9
    on_point: boolean
        If True, the circles will bounce on their centers when encountering a polygon edge.
        If False, they will bounce on their edges.
        Default is True.
    polygons: Polygons class
        Special numba class type containing the polygon shapes which will contain the circles.
    n : integer
        The amount of circles.
        Determined on class object initialization.
    vx : 1D numpy array, floats
        Velocity over the x-axis. 
        Determined during circlepack.run(). Initializes as random uniform between -0.01 and 0.01.
    vy : 1D numpy array, floats
        Velocity over the y-axis.
        Determined during circlepack.run(). Initializes as random uniform between -0.01 and 0.01.
    dx : 2D numpy array, floats
        Distance between two circles over the x-axis. Determined during initialization.
    dy : 2D numpy array, floats
        Distance between two circles over the y-axis. Determined during initialization.
        
    """
    def __init__(self,  x, y, r, polygons, on_point = True, precision = 9):
        """
        Initialize the CirclePack class object.
        
        Thise class functions as a consistant storage of parameters.
                
        Parameters
        ----------
        r : 1D numpy array, floats
            Necesary to create class object, inititated on class creation.
        x : 1D numpy array, floats
            The x-coordinates of the circle centres.
            Necesary to create class object, inititated on class creation. 
        y : 1D numpy array, floats
            The y-coordinates of the circle centres.
            Necesary to create class object, inititated on class creation.
        polygons: Polygons class
            Special numba class type containing the polygon shapes which will contain the circles.
            Necesary to create class object, inititated on class creation.
        """
        self.x = x
        self.y = y
        self.r = r
        self.polygons = polygons
        self.on_point = on_point
        self.precision = precision
        self.n = len(self.x)
        self.d = np.zeros((self.x.size, self.x.size))
        self.dx = np.zeros((self.x.size, self.x.size))
        self.dy = np.zeros((self.x.size, self.x.size))
        self.vx = np.random.uniform(-1, 1, size = self.n)/100
        self.vy = np.random.uniform(-1, 1, size = self.n)/100
    
    def set_precision(self, new_precision):
        self.precision = new_precision
        
    @property
    def circle_indeces(self):
        """
        Returns
        -------
        1D numpy array, int
            The indexes of all the circles.
        """
        return np.array(list(range(self.n)))
    
    @property
    def fp(self):
        """
        fp = future positions
        The points the circles will be in after adding their velocity.

        Returns
        -------
        2D numpy array, float
            A CirclePack.n by 2 numpy array whith cartesian coordinates of the future circle center locations
        """
        ball_fp = np.column_stack((self.x + self.vx, self.y + self.vy))
        return NbRound2(ball_fp, self.precision)
    
    def fp_ball(self, ball):
        """
        fp = future positions
        Returns the future position of a specific circle

        Parameters
        ----------
        ball : int
            Index of the circle of which the future position is requested.

        Returns
        -------
        1D numpy array
            Cartesian coördinates of the future ball position.
        """
        ball_fp = np.array((self.x[ball] + self.vx[ball], self.y[ball] + self.vy[ball]))
        return NbRound1(ball_fp, self.precision)
    
    @property
    def p(self):
        """
        p = positions
        THe cartesian coordinates of the circle positions.

        Returns
        -------
        2D numpy array, float
            A CirclePack.n by 2 numpy array whith cartesian coordinates of the circle center locations
        """
        ball_p = np.column_stack((self.x , self.y))     
        return NbRound2(ball_p, self.precision)
    
    def p_ball(self, ball):
        """
        p = positions
        Returns the position of a specific circle

        Parameters
        ----------
        ball : int
            Index of the circle of which the position is requested.

        Returns
        -------
        1D numpy array
            Cartesian coördinates of the ball position.
        """
        ball_p = np.array((self.x[ball], self.y[ball]))
        return NbRound1(ball_p, self.precision)
    
    def update_positions(self):
        """
        Sets the x- and y-coordinates based on current locations and their velocities.

        Returns
        -------
        None.

        """
        self.x = NbRound1(self.x + self.vx, self.precision)
        self.y = NbRound1(self.y + self.vy, self.precision)
        
    def CheckEdges(self):
        """
        Check if the circles intersect with their polygon.

        Returns
        -------
        edge_check : 3D numpy array, boolean.
            For each circle is checked if it overlaps with a polygon and on which segment this occurs.

        """
        check_int = np.zeros((self.n, self.polygons.n, self.polygons.polygons.shape[1]))
        edge_check = check_int > 0
        for b in range(self.n):
            for s in range(self.polygons.n):
                for seg in range(np.logical_not(np.isnan(self.polygons.polygons[s][:,0])).sum() - 1):
                    edge_check[b][s][seg] = CheckCircleIntersect(self.fp_ball(b), self.r[b], self.polygons.polygons[s][seg], self.polygons.polygons[s][seg+1])
        return edge_check
    
    def Distance(self):
        """ 
        Calculates the distances between points
        
        Returns
        -------
        d : 2D numpy array, floats
            Cartesian distance between the circle centres
        dx : 2D numpy array, floats
            The distance between the circle centres, over the x-axis
        dy : 2D numpy array, floats
            The distance between the circle centres, over the y-axis.
        """
        # returns a numpy array with the distance between all the balls
        dx = np.zeros((self.n, self.n))
        dx[:] = self.x
        self.dx = dx - dx.T
        
        dy = np.zeros((self.n, self.n))
        dy[:] = self.y
        self.dy = dy - dy.T
        self.d = np.sqrt(self.dx**2 + self.dy**2)
        self.Overlap()
        return self.d, self.dx, self.dy
    
    def Overlap(self):
        """
        Sets (ans returns) CirclePack.dr.

        Returns
        -------
        dr  : 2D numpy array, boolean
            A CirclePack.n by Circlepack.n matrix with boolean values, showing if a circle on a row overlaps with a circle on the columns (True) or not (False).
        """
        dr = np.zeros((self.n, self.n))
        dr[:] = self.r
        dr = self.d - dr - dr.T
        np.fill_diagonal(dr, np.max(self.r) + 999) # set distances with self to huge
        self.dr = dr < 0
        return self.dr

    def GetForce(self, move):
        """
        When circles overlap, they apply a force on each other to push themselves away.

        Parameters
        ----------
        move : 1D numpy array
            Indeces of the circles that overlap and therefore need to be moved.

        Returns
        -------
        fx : 1D numpy array, floats
            Force to be applied on the x-axis.
        fy : 1D numpy array, floats
            Force to be applied on the y-axis.

        """
        fx = np.zeros(self.n)
        fy = np.zeros(self.n)
        for b in move:
            # The balls it is overlapping w/
            ob = np.where(self.dr[b])[0]
            
            # The distances
            diff_x = self.dx[b][ob]
            diff_y = self.dy[b][ob]
            diff_xy = np.column_stack((diff_x, diff_y)) # The distance between the balls in vectors
            
            # Normalize the distance to vector with length of 1
            diffn = Normalize(diff_xy)
            
            # Set force as a function of the absolute distance to other circles
            diff_dist = self.d[b][ob]
            
            force1 = -1 * diffn / (1 / (diff_dist**2)).reshape(diffn.shape[0], 1)
            
            # Check for norm bounds
            normbi = np.zeros((force1.shape[0], 2))
            for i, vectorbi in enumerate(force1):
                normbi[i] = np.linalg.norm(vectorbi)
            normbibool = np.where(normbi > 0)[0]
            force = force1.copy()
            for ni in normbibool:
                force[ni] = np.subtract(force1[ni], np.array([self.vx[b], self.vy[b]]))
            net_force = force.sum(axis = 0)/(self.n - 1)
            
            if np.linalg.norm(net_force) < self.r[b]/10:
                net_force = Normalize(AddAxis(net_force)) * self.r[b] / 10
                net_force = net_force[0]
            
            # Set force
            fx[b] = net_force[0] 
            fy[b] = net_force[1] 
        return fx, fy
    
    def BouncePointCircle(self, edge_detect):
        """
        Returns the average of all the line segments the circles overlaps with.
        TODO: It doesn't need to return a value for all the circles.
    
        Parameters
        ----------
        edge_detect : 1D numpy array (C.n)
            Boolean array with True for the circles that overlap, False for those wo do not.
    
        Returns
        -------
        1D numpy array floats (len(np.unique(edge_detect[0])), 2)
            The average of all the line segments the circles overlaps with..
        """
        bounce_points = np.full((self.n, self.polygons.n, self.polygons.polygons.shape[1], 2), np.nan)
        for i, ball in enumerate(edge_detect[0]):
            p = edge_detect[1][i]
            seg = edge_detect[2][i]
            cx, cy, dx, dy, dr, big_d, discriminant = CircleLineIntersectDiscriminant(self.fp_ball(ball), 
                                                                                      self.r[ball], 
                                                                                      self.polygons.polygons[p][seg], 
                                                                                      self.polygons.polygons[p][seg + 1])
            intersections = FullLineIntersections(cx, cy, dx, dy, dr, big_d, discriminant)
            bounce_points[ball, p, seg] = Mean0(intersections)        
        bounce_point = NanMean12(bounce_points)        
        return bounce_point[np.unique(edge_detect[0])]
    
    def BouncePoint(self):  
        """
        Calculates the representative velocity of the bounce with the walls. Circles "bounce" on the circle centre.
        
        Sets the velocity of the balls to stay within the polygon.
        """           
        # Express the path the circle will travel as a line (lp)
        p = self.p # current positions
        fp = self.fp # future ball position
        lp = np.zeros((self.n, 2, 2))
        for i in range(self.n):
            lp[i][0] = p[i]
            lp[i][1] = fp[i]
       
        # Determine which circles will end up outside the polygon and therefore will have to bounce
        bouncy = np.where(np.logical_not(self.polygons.ContainsPoints(fp)))[0]
        if len(bouncy) > 0:
            bouncy_lines = lp[bouncy]
            
            # Determine where the traveling path lp intersects the line and with which segment.
            intersect, intersegments = self.polygons.PolygonIntersection(bouncy_lines)
            
            # Determine where the ball would bounce towards and set velocity to move towards that point
            self.x[bouncy] = intersect[:, 0]
            self.y[bouncy] = intersect[:, 1]
            v = np.transpose(np.vstack((self.vx, self.vy)))
            v[bouncy] = MirrorPoints(fp[bouncy], intersegments) - intersect
            self.vx, self.vy = np.transpose(v)
            
    def BounceEdge(self):
        """
        Calculates the representative velocity of the bounce with the walls. Circles "bounce" on the circle edge.
        
        Sets the velocity of the balls to stay within the polygon.
        """    
        # Determine if balls touch the edges
        p = self.p  # current positions
        fp = self.fp # future ball position
        bp = fp.copy()
        edge_check = self.CheckEdges() # which balls intersect with which polygon, on which segment(s)
        pre_edge_detect = np.argwhere(edge_check)
        edge_detect = pre_edge_detect.T
        bouncy = np.unique(edge_detect[0]) # The balls that intersect should move accordingly
        if len(bouncy) > 0:
            
            # Get the point from which the circles bounces (only an approximation on non-straight lines)
            bp[bouncy] = self.BouncePointCircle(edge_detect) 
            
            # Get the vector between that point and the future ball position
            bounce_vector = np.subtract(bp, fp) 
            
            # Get the vector of the line between the radius and the centre of the circle, along the vector with the bounce point
            radius_vector = np.zeros_like(bounce_vector)
            for i in bouncy:   
                radius_vector[i] = bounce_vector[i]*self.r[i]/np.linalg.norm(bounce_vector[i])
            
            
            # get the difference between the two vector to get the vector along which to move the ball to avoid overlap with polygon
            bounce = 2 * (bounce_vector - radius_vector)
            fpbounce = fp + bounce
            self.vx, self.vy = (fpbounce - p).T

    def set_v_to_r(self):
        """
        Set max speed (self.vx, self.vy) to radius of circle. This way, when on_point = False, 
        the circle is less likely to overshoot the polygon boundary.
        """
        vr = np.column_stack((self.vx, self.vy))
        vn = np.zeros(self.n)        
        for i, vri in enumerate(vr):
                vn[i] = np.linalg.norm(vri)
                
        set_v = np.where(vn > self.r)[0]
        self.vx[set_v] *= self.r[set_v]/vn[set_v]
        self.vy[set_v] *= self.r[set_v]/vn[set_v]
        self.vx = NbRound1(self.vx, self.precision)
        self.vy = NbRound1(self.vy, self.precision)  
        
    def run(self):
        """
        Calculates forces between balls. Then determines if they should bounce against the surroundin polygon feature.
        Check if they don't escape the polygon none the less (in case of double bounces) and adjust speed vector to point towards 
        CirclePack.polygons.centringpoint.

        Returns
        -------
        None.

        """
        # determine the distances between the circles
        self.Distance()        
        
        # The circles that should move, and shouldn't
        move = np.where(Any1(self.dr))[0] # On the index are the balls that are checked upon
        no_move = NotInInt(self.circle_indeces, move) # The balls that shouldn't
        
        # Stop the circles that don't have to move and appl force to those that do.
        self.vx[no_move] = 0
        self.vy[no_move] = 0
        fx, fy = self.GetForce(move)
        
        # Velocity because of forces between balls
        self.vx = NbRound1(self.vx + fx, self.precision)
        self.vy = NbRound1(self.vy + fy, self.precision)
        
        # Make sure the circles do not move too fast. When passing the polygon some might behave differently on bouncing.
        self.set_v_to_r()   
        
        # Determine velocity due to bouncing
        if self.on_point:
            self.BouncePoint()
        else:
            self.BounceEdge() 
             
        # The balls can for some reason still be outside at times. Bring them back home.
        ball_fp = self.fp  
        ball_io = np.where(np.logical_not(self.polygons.ContainsPoints(ball_fp)))[0]
        if len(ball_io) > 0:
            diff_v = NbRound2(self.polygons.centringpoint - ball_fp[ball_io], self.precision)
            self.vx[ball_io] = diff_v[:,0]
            self.vy[ball_io] = diff_v[:,1]
    
            self.set_v_to_r() 
         
        # Move
        self.update_positions()        
        return
    
    def check(self):
        """
        Determine if the circles are not moving, are inside the desired area
        and do not still overlap.

        Returns
        -------
        check : bool
            True if the circles are not moving, are inside the desired area
            and do not still overlap. False if any of them are the opposite.
        list
            To check which one is the culprit.

        """
        vel = self.check_vel() #True when moving
        box = np.invert(self.check_bounds()) # Geeft True wanneer het buiten de box is
        dis = self.check_overlap() # Geeft True wanneer er cirkels overlappen
     
        check = np.array([vel, box, dis]).any() #wanneer dit True returned, is het dus nog niet goed!
        return check, [vel, box, dis]
    
    def check_vel(self):
        """
        Check if the balls are still moving.

        Returns
        -------
        bool
            True if still moving, Flase if not.

        """
        return np.array([self.vx.any(), self.vy.any()]).any()
     
    def check_bounds(self):
        return self.polygons.ContainsPoints(self.p).all() # Returns True when all are inside
    
    def check_overlap(self):
        return self.dr.any() # True when overlap
      
    def run_till_check(self, max_itt):
        itt = 0
        self.run()
        while itt < max_itt and self.check_vel():
            self.run()
            itt+=1
        return itt
