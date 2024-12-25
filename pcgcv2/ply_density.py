import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from data_utils import read_ply_ascii_geo


def compute_density(points, k=5):
    # Create a KDTree for efficient nearest neighbor queries
    tree = KDTree(points)

    # Query the k-th nearest neighbor for each point
    # We use k+1 because the nearest neighbor includes the point itself at distance 0
    distances, _ = tree.query(points, k=k + 1)

    # Get the distance to the k-th nearest neighbor
    kth_distances = distances[:, k]

    # Avoid division by zero; small epsilon value
    epsilon = 1e-9
    densities = 1.0 / (kth_distances ** 3 + epsilon)

    return densities


# Example usage:
# Assuming `points` is your N x 3 tensor loaded from the .ply file.
# points = np.random.rand(1000, 3)  # Example: 1000 random 3D points

path = '/media/student/e15ac0f9-6b08-41c6-b970-5f45828d563b/point_cloud_datasets/testdata/MVUB/andrew_vox9_frame0000.ply'
# path = '/media/student/e15ac0f9-6b08-41c6-b970-5f45828d563b/point_cloud_datasets/testdata/MVUB/david_vox9_frame0000.ply'
# path = '/media/student/e15ac0f9-6b08-41c6-b970-5f45828d563b/point_cloud_datasets/testdata/MVUB/phil_vox9_frame0139.ply'
# path = '/media/student/e15ac0f9-6b08-41c6-b970-5f45828d563b/point_cloud_datasets/testdata/MVUB/sarah_vox9_frame0023.ply'

data = read_ply_ascii_geo(path)

densities = compute_density(data, k=5)
print(densities)


# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot where color is based on the computed densities
scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=densities, cmap='viridis', marker='.')

# Adding a color bar to indicate density levels
color_bar = fig.colorbar(scatter, ax=ax)
color_bar.set_label('Density')

# Setting labels and title
ax.set_title('3D Point Cloud Density Visualization')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')

plt.show()
print('---end---')