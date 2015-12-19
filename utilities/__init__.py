import numpy
import matplotlib.pyplot as plt

import random, math
from collections import namedtuple
from random import random, uniform

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def Euler_distance(a, b):
    return numpy.sqrt(numpy.sum(numpy.square(a - b)))

def Euler_distance_square(a, b):
    return numpy.sum(numpy.square(a - b))

def generate_half_moon(n = 1000, radius=10, width=10, distance_x=7.5, distance_y=-2.5):
    r = radius
    w = width
    dx = distance_x
    dy = distance_y
    
    positive_list = []
    negative_list = []
    
    Point = namedtuple('Point', ('x', 'y'))
    
    def __generate_point_uniformly(radius, width, distance_x, distance_y, positive):
        # reference: http://stackoverflow.com/questions/9048095/create-random-number-within-an-annulus
        
        t = math.pi * random()
        if positive:
            pass
        else:
            t = -t
        
        R_square = (radius + width) ** 2
        r_square = radius ** 2
        r = math.sqrt((R_square - r_square) * random() + r_square)
        
        x = r * math.cos(t) 
        y = r * math.sin(t)
        
        if positive:
            x -= distance_x
            y += distance_y
        else:
            x += distance_x
            y -= distance_y
        
        return Point(x, y)
    
    for i in range(n):
        positive_list.append(__generate_point_uniformly(r, w, dx, dy, True))
        negative_list.append(__generate_point_uniformly(r, w, dx, dy, False))
        
    return ((positive_list, (1, 0)), (negative_list, (0, 1)))

def plot_half_moon(half_moon):
    p_x_list = [point.x for point in half_moon[0][0]]
    p_y_list = [point.y for point in half_moon[0][0]]
    n_x_list = [point.x for point in half_moon[1][0]]
    n_y_list = [point.y for point in half_moon[1][0]]
    
    fig = plt.figure()
    axes = fig.add_subplot(1, 1 ,1)
    axes.set_aspect('equal', 'datalim')
    s1 = axes.scatter(p_x_list, p_y_list, c='blue', marker='+')
    s2 = axes.scatter(n_x_list, n_y_list, c='red', marker='x')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_title('Sample half_moon')
    fig.show()