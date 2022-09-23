from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_voxels(voxel_array):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.voxels(voxel_array, edgecolor=None)

    plt.show()

import argparse
parser = argparse.ArgumentParser(description='program to visualize voxel teapots w/ matplotlib')
parser.add_argument("-f", "--file_name", help=".npy file to visualize", nargs='*')
args = parser.parse_args()
filename = Path(args.file_name[0])
with open(filename, 'rb') as f:
    voxels = np.load(f)
    if len(voxels.shape) > 3:
        print(voxels)
        voxels = np.squeeze(voxels, axis=3)
        trim = voxels < .9
        voxels[trim] = 0
    visualize_voxels(voxels)
