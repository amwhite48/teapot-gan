import numpy as np
from visualize import visualize_voxels
import random

# generate a teapot in a 3d np array
# function for defining a teapot - 3 possible body shapes (round, royal, kettle)
# add a handle - curve can be skewed to one side
# add a little sphere/cylinger thing on the top
# 0 means nothing there, 1 
w = 64
min_rad = 0.5
max_rad = 0.8
min_scale = 0.1
max_scale = 0.3

def generate_teapot():
    teapot_array = np.zeros((64,64,64))
    

    radius = random.uniform(min_rad, max_rad)
    point = np.array([0,0,0])
    height_scale_factor = min_scale + random.random() * (max_scale - min_scale)


    for i in range(w):
        for j in range(w):
            for k in range(w):
                if in_sphere(i, j, k, point, radius, height_scale_factor):
                    teapot_array[i,j,k] = 1.0           

    return teapot_array

# map a number 
def map_to_space(num):
    return (num - 32) / 32

def in_sphere(x, y, z, point, rad, height_scale):
    diff = np.array([map_to_space(x) - point[0], map_to_space(y) - point[1], (map_to_space(z) - point[2])*(1+height_scale) + rad/2])
    dist = np.sum(np.power(diff, 2))
    return dist < rad ** 2

def in_hemisphere(x, y, z, point, rad, height_scale):
    diff = np.array([map_to_space(x) - point[0], map_to_space(y) - point[1], (map_to_space(z) - point[2])*(1-height_scale) + rad])
    dist = np.sum(np.power(diff, 2))
    return dist < rad ** 2    


visualize_voxels(generate_teapot())
# teapot body - vary height, width

# round body - flat sphere

# royal body - wide bottom, skinnier top

# kettle body - half sphere


# teapot handle

# semicircle handle

# half heart handle


# teapot top - vary cylinder vs sphere

# teapot spout - vary length, width
# on top half of teapot, 