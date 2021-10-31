import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from circlepack import CirclePack
from polygon import Polygon, make_numba_array

def grow(C, max_itt, grow_rate = 1.00001):
    print('Grow initiated')
    log_x = [] # Store the locations and radii of the circles for image
    log_y = []
    log_r = []
    itt = 0
    frames = 0
    c = True
    while c:
        frames +=1
        itt = C.run_till_check(max_itt) # Optimize the distribution of the circles in the container
        print(f"Frame: {frames}, radius: {C.r[0]}, after {itt} itterations")
        log_x.append(C.x.copy()) 
        log_y.append(C.y.copy())
        log_r.append(C.r.copy())
        if C.check()[0] or itt >= max_itt: # Make sure the packing is succesful or within the amount of maximum iteration. 
            return log_x, log_y, log_r
        C.r = np.round(C.r*grow_rate , C.precision) # When the packing has been succesful, initate grow
        
    if len(log_x) > 1:
        return log_x, log_y, log_r
    else:
        raise(Exception("With these amount, radii and maximum iterations, no solution was found. Try a smaller starting radius or higher amount of iterations."))

def plot_pack(C):
    ball_x = C.x
    ball_y = C.y
    ball_r = C.r
    f, ax = plt.subplots()
    for i, pol in enumerate(C.polygons.polygons):

        xi,yi = zip(*pol)
        ax.plot(xi,yi, 'k')
    ax.axis('scaled')
    plt.axis('off')


    for c in range(len(ball_x)):
        ball = plt.Circle((ball_x[c], ball_y[c]), radius=ball_r[c], picker=True, fc='none', ec='k')
        ax.add_patch(ball)  

SCALE = 10 # The The algorithm performs best with circles between 0.5 and 5.
balls_n = 64
balls_r = 0.03 * SCALE
    
poly_edge = np.array([(0,0), (0,1*SCALE), (1*SCALE,1*SCALE), (1*SCALE,0), (0,0)]) # The polygon, closed (where the last point in the polygon is the same as the first.
poly = [poly_edge] # There can be multiple polygons in the packing, with different lengths

poly = make_numba_array(poly) # Make sure they have the same lengths and can be used by numba
poly = Polygon(poly) # Initiate the container class

x_, y_ = poly.centringpoint # Get the midpoint (unless it falls outside the container, then it is a random point in it)
ball_r = np.ones(balls_n) * balls_r
ball_x = np.ones(balls_n) * x_
ball_y = np.ones(balls_n) * y_

C = CirclePack(ball_x, ball_y, ball_r, poly, on_point = False, precision = 7) # Initiate the circle pack

max_itt = 10000 # Set the maximum number of iterations. Higher takes longer, but is more likely to find solutions to higher circle radii
log_x, log_y, log_r = grow(C, max_itt, grow_rate = 1 + 1e-2) # Lower grow_rate makes sure the circles are packed sooner, but is more likely to suboptimal.
C.x = log_x[-2] # The final one is not an optimal configuration
C.y = log_y[-2]
C.r = log_r[-2]

plot_pack(C)

#%% A cell to convert the logs to a gif
fig = plt.figure()
for p in C.polygons.polygons:
    xi, yi = zip(*p)
    xi, yi = np.array(xi), np.array(yi)

    plt.plot(xi,yi, 'k')
plt.axis('scaled')
plt.axis('off')

def gif_log(look, log_x, log_y, log_r):
    patch = []
    
    fig.clf()
    
    for p in C.polygons.polygons:
        xi, yi = zip(*p)
        xi, yi = np.array(xi), np.array(yi)
    
        plt.plot(xi,yi, 'k')
    plt.axis('scaled')
    plt.axis('off')
   
    for c in range(balls_n):
        ball = plt.Circle((log_x[look][c], log_y[look][c]), radius=log_r[look][c], picker=True, fc='none', ec='k')
        patch.append(plt.gca().add_patch(ball))
    
    print(f'Frame number: {min(len(log_x), look)}')
    return patch    
    
anim = animation.FuncAnimation(fig, gif_log, frames=len(log_x)-1, fargs = [log_x, log_y, log_r], interval=100, repeat = False, blit = True)
anim.save('circlepack.gif', dpi=160, writer='pillow')     
