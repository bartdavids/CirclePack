# CirclePack
Controlled circle packing in a container.
Based on this blog post:

http://www.codeplastic.com/2017/09/09/controlled-circle-packing-with-processing/#comment-22

With added boundaries and edge detection, written in python and sped up using numba.

You can choose for the circles to bounce on there edge, or on their circle center.

There are 5 .py modules
- circlepack.py, storing the CirclePack class. This class stores the circles, the Polygon class and the optimization algorithm.
- polygon.py, storing the container class Polygon
- numba_utils.py, where numba-friendly alternatives to numpy functions are stored
- linalg_utils.py, where some miscelaneous functions for circle, line and point interaction are stored
- circlepack_example.py, where an example is presented. This examples distributes the circles as efficiently as possible in the container, growing them till they do not fit any longer. It creates a .gif with the progress at points of growth.

## Some notes
Circle packing is not an easy problem. Most methods I encountered did not adress the problem of a container, or a hole in the container, or took extremely long to solve the problem. However, my dive into the subject was brief and not with the intention to make the best possible way to solve this problem. It is for me personally a study into numba and has some overlap with origami design, another hobby of mine.

The method I use to determine the bounce vector is graphically shown in the "edge detect method.png". The points where the circle intersects (red x's) are averaged (blue x) to determine the bounce vector. It is a short and quick way to handle it, and much easier to handle with numba. However, it is not an exact way and does not take into consideration the velocity vector.
