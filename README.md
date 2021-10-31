# CirclePack
Controlled circle packing in a container.
Based on this blog post:

http://www.codeplastic.com/2017/09/09/controlled-circle-packing-with-processing/#comment-22

With added boundaries and edge detection, written in python and sped up using numba.

You can choose for the circles to bounce on their edge, or on their circle center.

There are 5 .py modules
- circlepack.py, storing the CirclePack class. This class stores the circles, the Polygon class and the optimization algorithm.
- polygon.py, storing the container class Polygon
- numba_utils.py, where numba-friendly alternatives to numpy functions are stored
- linalg_utils.py, where some miscelaneous functions for circle, line and point interaction are stored
- circlepack_example.py, where an example is presented. This examples distributes the circles as efficiently as possible in the container, growing them till they do not fit any longer. It creates a .gif with the progress at points of growth.

## Note
Circle packing is not an easy problem. Most methods I encountered did not adress the problem of a container, or a hole in the container, or took extremely long to solve the problem. However, my dive into the subject was brief and not with the intention to make the best possible way to solve this problem. It is for me personally a study into numba.

## Dependancies
This code has been made with numpy (1.19.2) and numba (0.54). For the example matplotlib is used as well to show the circle and container configuration.

## Process
When the optimization process is started, the circles will look for the first empty space they can find. Below a simple circle packing example is taken, where 64 circles of equal radii are in a square 1x1 container:

![image](https://github.com/bartdavids/CirclePack/blob/main/Images/Process%20-%20run.gif)

In the example, an addition function is used to find the largest radius the specified amount of circles can be inside the container. After the optimization has run untill the circles do not need to move to stop overlapping with their neighbours or stay inside the container, they grow. After that grow the process is started again until there is no more room to grow (a certain amount of itterations is reached). In the below figure this process is shown by taking a snapshot the frame just before a growth spurt is initated.

![image](https://github.com/bartdavids/CirclePack/blob/main/Images/Process%20-%20grow.gif)

## Assumptions
The method I use to determine the bounce vector is graphically shown here:

![image](https://github.com/bartdavids/CirclePack/blob/main/Images/Edge%20detect%20method.JPG)

The points where the circle intersects (red x's) are averaged (blue x) to determine the bounce vector. It is a short and quick way to handle it, and much easier to handle with numba. However, it is not an exact way and does not take into consideration the velocity vector. In the first video example this choice is shown to have consequences, where circles may bounce with more vigor from the corners due to the additional "overlap" (the blue x) compared to the actual overlap (with the red x's).

## Perfomance - results
The results of the optimization have been briefly compared to the results presented at http://www.packomania.com/. On this website the benchmark bests for different containers and amount of circles are stored.

In the above example of the growing case, the largest radii of the circles is approximated with this method. This method approximates the circle configuration forthe 64 circle case at Packomania and has a radius of 0.06230611, 98.18% of the circle radius presented at Packomania for the 64 circles in a square situation. After some tweaking of the parameters the result can be enhanced. My personal best on the 64 circle problem has been a radius of 0.06301875, surpassing 99% approximation of the Packomania benchmark.

## Performance - time
Numbafying this code gave it a little over 10x speed boost, making a higher precision and approximation to the Packomania benchmarks possible and more fun.
