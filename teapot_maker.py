import numpy as np
from visualize import visualize_voxels
import random
import math

# generate a teapot in a 3d np array
# function for defining a teapot - 3 possible body shapes (round, royal, kettle)
# add a handle - curve can be skewed to one side
# add a little sphere/cylinger thing on the top
# 0 means nothing there, 1 
w = 16
min_rad = 0.5
max_rad = 0.7
min_scale = 0.1
max_scale = 0.3

def generate_teapot():
    teapot_array = np.zeros((16,16,16))
    sphere_pot = random.random() < 0.5
    semicircle_handle = random.random() < 0.5
    

    radius = random.uniform(min_rad, max_rad)
    point = np.array([0,0,0])
    height_scale_factor = min_scale + random.random() * (max_scale - min_scale)
    handle_height_scale = random.uniform(0.2, 0.25)
    
    
    if semicircle_handle:
        height_offset = 0
    else:
        height_offset = random.uniform(.2, .3)
    # spout top always around top of teapot
    if sphere_pot:
        spout_top = [-radius - .2,0,radius/2] 
        spout_bottom_z = random.uniform(-radius, -radius/2)
        top_handle_z = radius/2 - .1    
        handle_top = [radius, 0, handle_height_scale*(1+height_scale_factor) - radius / 2 + height_offset]
        handle_bottom = [radius, 0, -handle_height_scale*(1+height_scale_factor) - radius / 2 + height_offset]
    else:
        spout_top = [-radius - .2,0,.1]
        top_handle_z = -.05
        spout_bottom_z = random.uniform(-radius+.2, -radius)
        handle_top = [radius - .1, 0, handle_height_scale*(1+height_scale_factor) - radius / 1.5 + height_offset]
        handle_bottom = [radius - .1, 0, -handle_height_scale*(1+height_scale_factor) - radius + height_offset]
    # randomize the height later
    spout_bottom = [-radius + .2,0,spout_bottom_z]


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
                if in_semicircle_handle([i, j, k], handle_top, handle_bottom, .04):
                    teapot_array[i,j,k] = 1.0 
                elif in_spout([i, j, k], spout_top, spout_bottom, .025):
                    teapot_array[i,j,k] = 1.0 
                elif in_top([i,j,k], top_handle_z, .12):
                    teapot_array[i,j,k] = 1.0 
    return teapot_array

# map a number 
def map_to_space(num):
    return (num - 8) / 8

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

def in_spout(point, top, bottom, thickness):
    in_space = [map_to_space(point[0]), map_to_space(point[1]), map_to_space(point[2])]
    dist = distance(top, in_space) + distance(bottom, in_space)
    line_length = distance(top, bottom)
    return line_length - thickness < dist < line_length + thickness and in_space[2] < top[2] - thickness and in_space[2] > bottom[2] + thickness

# top is a cylinder
def in_top(point, top_z, radius):
    in_space = [map_to_space(point[0]), map_to_space(point[1]), map_to_space(point[2])]
    return math.sqrt(math.pow(in_space[1], 2) + math.pow(in_space[0], 2)) < radius and in_space[2] > top_z and in_space[2] < top_z + radius

def distance(p, q):
    return math.sqrt(math.pow(p[0] - q[0], 2) + math.pow(p[1] - q[1], 2) + math.pow(p[2] - q[2], 2))

# visualize_voxels(generate_teapot())
# save 
def generate_teapot_dataset(num_teapots):
    for i in range(num_teapots):
        teapot = generate_teapot()
        filename = "teapot_data/teapot_ex_" + str(i) + ".npy"
        with open(filename, 'wb') as f:
            np.save(filename, teapot)


generate_teapot_dataset(5000)
