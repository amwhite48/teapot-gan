import numpy as np
from visualize import visualize_voxels
import random
import math

# generate a teapot in a 3d np array
# function for defining a teapot - 3 possible body shapes (round, royal, kettle)
# add a handle - curve can be skewed to one side
# add a little sphere/cylinger thing on the top
# 0 means nothing there, 1 
w = 64
min_rad = 0.5
max_rad = 0.7
min_scale = 0.1
max_scale = 0.3

def generate_teapot():
    teapot_array = np.zeros((64,64,64))
    

    radius = random.uniform(min_rad, max_rad)
    point = np.array([0,0,0])
    height_scale_factor = min_scale + random.random() * (max_scale - min_scale)
    handle_height_scale = random.uniform(0.2, 0.25)
    # if heart handle, move it higher
    height_offset = 0 #random.uniform(0, 0.5)
    sphere_pot = True
    semicircle_handle = True

    if sphere_pot:
        handle_top = [radius, 0, handle_height_scale*(1+height_scale_factor) - radius / 2 + height_offset]
        handle_bottom = [radius, 0, -handle_height_scale*(1+height_scale_factor) - radius / 2 + height_offset]
        

    for i in range(w):
        for j in range(w):
            for k in range(w):
                if sphere_pot:
                    if in_sphere(i, j, k, point, radius, height_scale_factor):
                        teapot_array[i,j,k] = 1.0
                        
                else:
                    if in_hemisphere(i, j, k, point, radius, height_scale_factor):
                        teapot_array[i,j,k] = 1.0 
                # calculate handle top and bottom based off height scale factor
                      
                if in_semicircle_handle([i, j, k], handle_top, handle_bottom, .03):
                    teapot_array[i,j,k] = 1.0 
    return teapot_array

# map a number 
def map_to_space(num):
    return (num - 32) / 32

def in_sphere(x, y, z, point, rad, height_scale):
    diff = np.array([map_to_space(x) - point[0], map_to_space(y) - point[1], (map_to_space(z) - point[2])*(1+height_scale) + rad/2])
    dist = np.sum(np.power(diff, 2))
    return dist < rad ** 2

# for half-circle type teapots
def in_hemisphere(x, y, z, point, rad, height_scale):
    diff = np.array([map_to_space(x) - point[0], map_to_space(y) - point[1], (map_to_space(z) - point[2])*(1-height_scale) + rad])
    dist = np.sum(np.power(diff, 2))
    return dist < rad ** 2    

# teapot handle
# semicircle handle
def in_semicircle_handle(point, top, bottom, thickness):
    # near border of semicircle 
    # top and bottom are top and bottom points of handle
    radius = (top[2] - bottom[2])/2
    center = (np.array(top) + np.array(bottom)) / 2
    diff = np.array([map_to_space(point[0]) - center[0], map_to_space(point[2]) - center[2]])
    dist = np.sum(np.power(diff, 2))
    return abs(map_to_space(point[1])) < thickness*3 and (radius ** 2 - thickness) < dist < (radius ** 2 + thickness) # and map_to_space(point[1]) 

def in_heart_handle(top, bottom):
    pass

visualize_voxels(generate_teapot())

# half heart handle


# teapot top - vary cylinder vs sphere

# teapot spout - vary length, width
# on top half of teapot, 