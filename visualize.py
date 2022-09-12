from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# N1 = 64
# N2 = 64
# N3 = 64
# ma = np.random.choice([0,1], size=(N1,N2,N3), p=[0.99, 0.01])

def visualize_voxels(voxel_array):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.voxels(voxel_array, edgecolor=None)

    plt.show()

