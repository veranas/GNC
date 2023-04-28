import numpy as np
from scipy.spatial.transform import Rotation
import plotly.graph_objs as go
import torch
import torch.linalg

def generate_points_on_cylinder(radius, height, num_points, theta_range=(0, 2 * np.pi)):
    """
    Generates a set of points randomly distributed around the surface of an open cylinder.

    Parameters:
        radius (float): the radius of the cylinder
        height (float): the height of the cylinder
        theta_range (tuple): a tuple (theta_min, theta_max) specifying the angular range of the cylinder
        num_points (int): the number of points to generate

    Returns:
        A matrix of shape (num_points, 3) containing the x, y, and z coordinates of the generated points.
    """

    # Generate random angles and heights
    theta = np.random.uniform(*theta_range, num_points)
    h = np.random.uniform(0, height, num_points)

    # Generate random radii that follow a square root distribution to ensure uniform distribution along the
    #  cylinder's surface
    r = radius

    # Calculate x, y, and z coordinates of the points
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = h

    # Combine the x, y, and z coordinates into a matrix
    points = np.column_stack((x, y, z))

    return points


def random_points_on_sphere(n_points, radius=1.0):
    """
    Generates n_points random points on the surface of a sphere.
    Returns a NumPy array of shape (n_points, 3).
    """
    # Generate n_points random values for the polar angle (theta) and azimuthal angle (phi)
    theta = np.random.uniform(0, np.pi, n_points)
    phi = np.random.uniform(0, 2 * np.pi, n_points)

    # Calculate the corresponding x, y, and z coordinates using the spherical coordinate formulae
    y = radius + np.sin(theta) * np.sin(phi) - radius
    z = radius + np.cos(theta) - radius
    x = radius + np.sin(theta) * np.cos(phi) - radius

    # Combine the x, y, and z coordinates into a NumPy array of shape (n_points, 3)
    points = np.vstack((x, y, z)).T

    return points


def normalize_point_cloud(points):
    """
    Normalizes a point cloud of shape (n, 3) to be within a unit square centered on the origin.
    Returns the normalized points as a NumPy array of the same shape.
    """
    # Find the minimum and maximum coordinates of the point cloud along each axis
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    # Calculate the scale factor needed to fit the point cloud within a unit square
    max_range = np.max(max_coords - min_coords)
    scale_factor = 1.0 / max_range

    # Scale the point cloud and shift it to be centered on the origin
    centered_points = scale_factor * (points - min_coords - (max_range / 2))

    return centered_points

def generate_pseudo_bunny(n_points):

    # Calculate 60% of n and assign it to variable a
    a = int(0.6 * n_points)

    # Calculate 40% of n / 2 and assign it to variables b and c
    b = c = int(0.4 * n_points / 2)

    # Check if the sum of a, b, and c equals n, and adjust c if necessary
    if a + b + c != n_points:
        a += n_points - (a + b + c)

    sphere = random_points_on_sphere(a, 1.0)

    cylinder_1 = generate_points_on_cylinder(radius=0.2, height=1.5, num_points=b, theta_range=(0.8 * np.pi, -0.8 * np.pi))
    cylinder_2 = generate_points_on_cylinder(radius=0.2, height=1.5, num_points=c, theta_range=(0.8 * np.pi, -0.8 * np.pi))

    cylinder_1 += np.expand_dims(np.array([0, 0, 1]), 0)
    cylinder_2 += np.expand_dims(np.array([0, 0, 1]), 0)

    # create a rotation matrix:
    angle_radians = np.radians(25)
    c = np.cos(angle_radians)
    s = np.sin(angle_radians)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, c, -s],
                                [0, s, c]])

    cylinder_1 = cylinder_1.dot(rotation_matrix)

    angle_radians = np.radians(-25)
    c = np.cos(angle_radians)
    s = np.sin(angle_radians)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, c, -s],
                                [0, s, c]])

    cylinder_2 = cylinder_2.dot(rotation_matrix)

    bunny = np.vstack((sphere, cylinder_1, cylinder_2))
    bunny = normalize_point_cloud(bunny)
    return bunny

def generate_impulse_noise(distance=1.0):
    """
       Implements a style of impulse noise or "salt and pepper" noise
       for simplicity this assumes that the noise is along the y-axis

    Parameters:
        distance (float): the +/- distance

    Returns:
        an x, y, z, array containing the noise vector
    """
    # Generate outlier noise by moving point along the random direction vector
    noise = np.array([0, 1, 0]) * distance * np.random.choice([-1, 1])

    return noise


def generate_spherical_noise(radius=5.0):
    """
        Spherical noise generated as in: H. Yang and L. Carlone, “A polynomial-time solution
         for robust registration with extreme outlier rates,” in Robotics: Science and Systems
         (RSS), 2019.

        Parameters:
            radius (float): the radius of the sphere describing the distribution

        Returns:
            a random x, y, z, array (float) uniformly distributed throughout the sphere
        """

    # Generate a random point in a cube with side length 2*radius
    x, y, z = np.random.uniform(-radius, radius, size=(3,))
    # Check if the point is inside the sphere
    while x ** 2 + y ** 2 + z ** 2 > radius ** 2:
        x, y, z = np.random.uniform(-radius, radius, size=(3,))
    return np.array([x, y, z])


def generate_rand_transform(distance=1.0):
    """
        Generates a random transform consisting of a random rotation and a random translation with fixed distance

        Parameters:
            distance (float): the distance of translation

        Returns:
            a randomly generated 3x4 transform as a numpy array
        """
    # Generate a random 3D direction vector
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)

    # Generate a random 3D rotation matrix
    rotation = Rotation.random().as_matrix()

    # Create a 4x4 transformation matrix
    transformation = np.identity(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = distance * direction
    transformation = transformation[:3, :]
    return transformation


def apply_outlier_noise(points, noise_function, percentage, magnitude=5.0, style="assign"):
    """
    Apply a given noise function to a randomly selected percentage of points in a given array of points.

    Parameters:
        points (list): The array of points to which the noise function should be applied.
        noise_function (function): The noise function to apply to the points.
        magnitude (float): The magnitude of the noise to add
        percentage (float): The percentage of points to apply the noise function to. Must be between 0 and 100.
        style (str): Style of noise addition choices are:
                "assign": replace the outlier with the noise value
                "add": add the value onto the outlier

    Returns:
        y_inliers (list): a one hot vector of all the inliers
        points (list): The updated array of points with the noise function applied.
    """
    n = points.shape[0]
    num_noise_points = int(n * percentage / 100)  # number of indices to select
    outlier_indices = np.random.choice(n, num_noise_points, replace=False)  # list of random indices
    y_outliers = np.zeros(n, dtype=int)  # one hot vector for random indices
    y_outliers[outlier_indices] = 1
    y_inliers = y_outliers ^ np.ones(y_outliers.shape, dtype=int)

    for index in outlier_indices:
        if style == "assign":
            points[index] = noise_function(magnitude)
        else:
            points[index] += noise_function(magnitude)

    return y_inliers, points


def plot_shapes(point_batches, colorscale, legends, title=""):
    # Concatenate the x,y,z coords together and define color assignments
    points = np.concatenate(point_batches, axis=0)

    trace_list = []
    for points, color, legend in zip(point_batches, colorscale, legends):
        trace_list.append(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers',
                                       name=legend, marker=dict(size=5, opacity=0.4, color=color)))

    # Create the layout for the plot

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(
                showticklabels=True,
                showgrid=True,
                zeroline=True
            ),
            yaxis=dict(
                showticklabels=True,
                showgrid=True,
                zeroline=True
            ),
            zaxis=dict(
                showticklabels=True,
                showgrid=True,
                zeroline=True
            )
        )
    )

    # Combine the trace and layout into a Figure object
    fig = go.Figure(data=trace_list, layout=layout)
    fig.show()


def weighted_MSE_cost(T, weights, source_points, target_points):
    """
    Calculate the weighted mean squared error cost between the transformed source points and the target points.

    Args:
        T (tensor): transformation matrix (4x4)
        weights (tensor): weights of each point (1D tensor)
        source_points (tensor): source points (Nx3 tensor)
        target_points (tensor): target points (Nx3 tensor)

    Returns:
        tensor: weighted mean squared error cost
    """
    # Transform the source points using the transformation matrix and calculate the squared distance
    # between the transformed source points and the target points.
    transformed_points = (T[:3, :3] @ source_points.T + T[:3, 3].unsqueeze(1)).T
    r_squared_weighted = (torch.sum((transformed_points - target_points) ** 2, dim=1) * weights)
    # Calculate the mean of the weighted squared distances.
    mse = torch.mean(r_squared_weighted, axis=0)
    return mse


def MSE_residuals(T, source_points, target_points):
    """
    Calculate the mean squared error residuals between the transformed source points and the target points.

    Args:
        T (tensor): transformation matrix (4x4)
        source_points (tensor): source points (Nx3 tensor)
        target_points (tensor): target points (Nx3 tensor)

    Returns:
        tensor: mean squared error residuals
    """
    # Transform the source points using the transformation matrix and calculate the squared distance
    # between the transformed source points and the target points.
    transformed_points = (T[:3, :3] @ source_points.T + T[:3, 3].unsqueeze(1)).T
    r = torch.sum((transformed_points - target_points) ** 2, dim=1)
    return r


def update_MSE_weights_GM(T, weights, c_bar, mu, source_points, target_points):
    """
    Update the weights of the points based on the mean squared error residuals and the tuning parameters.

    Args:
        T (tensor): transformation matrix (4x4)
        weights (tensor): weights of each point (1D tensor)
        c_bar (float): maximum expected error in the inliers
        mu (float): tuning parameter for the update rule, chosen large then reduced down to 1
        source_points (tensor): source points (Nx3 tensor)
        target_points (tensor): target points (Nx3 tensor)

    Returns:
        tensor: updated weights of each point (1D tensor)
    """
    # Compute the transformed source points using the transformation matrix.
    with torch.no_grad():
        transformed_points = (T[:3, :3] @ source_points.T + T[:3, 3].unsqueeze(1)).T
    r_sqrd = torch.sum((transformed_points - target_points) ** 2, dim=1)

    mu_C_sqrd = mu * c_bar ** 2

    # Calculate the updated weights using the update rule.
    weights = (mu_C_sqrd / (mu_C_sqrd + r_sqrd)) ** 2

    return weights


def update_MSE_weights_TLS(T, weights, c_bar, mu, source_points, target_points):
    """
    Updates weights for the total least squares (TLS) algorithm.

    Args:
        T: torch.Tensor of shape (4, 4). Current transformation matrix.
        weights: torch.Tensor of shape (N,). Current weights for each point.
        c_bar: float. Typically chosen as the maximum expected error in the inliers.
        mu: float. Reduction factor for mu chosen from a near zero increasing up to infinity to recover the original function
        source_points: torch.Tensor of shape (N, 3). Source points to be transformed.
        target_points: torch.Tensor of shape (N, 3). Target points to be matched with source points.

    Returns:
        torch.Tensor of shape (N, 1). Updated weights for each point.
    """

    # Compute current transformed points
    with torch.no_grad():
        transformed_points = (T[:3, :3] @ source_points.T + T[:3, 3].unsqueeze(1)).T

    # Compute squared residuals
    r_sqrd = torch.sum((transformed_points - target_points) ** 2, dim=1)

    # Reshape weights for broadcasting
    weights = weights.reshape(-1, 1)

    # Compute upper and lower bounds
    lb = mu * (c_bar ** 2) / (mu + 1)
    ub = (mu + 1) * (c_bar ** 2) / mu

    # Create masks for points above and below the bounds
    mask_above_ub = r_sqrd > ub
    mask_below_lb = r_sqrd < lb
    mask_between_ub_lb = (r_sqrd >= lb) & (r_sqrd <= ub)

    # Compute numerator for weights within bounds
    numerator = c_bar * torch.sqrt(mu * (mu + 1))

    # Update weights based on bounds and numerator
    weights[mask_below_lb] = 1
    weights[mask_above_ub] = 0
    weights[mask_between_ub_lb] = (numerator / (r_sqrd[mask_between_ub_lb]).reshape(-1, 1) ** 0.5) - mu

    return weights


def icp_transformation(source, target, weights):
    """
    Computes the transformation (rotation and translation) to align the source
    point cloud to the target point cloud using the Iterative Closest Point (ICP)
    algorithm.

    Arguments:
        source (Tensor): n x 3 tensor of source point cloud coordinates.
        target (Tensor): n x 3 tensor of target point cloud coordinates.
        weights (Tensor): n x 1 tensor of weights for the correspondences.

    Returns:
        R (Tensor): 3 x 3 rotation matrix.
        t (Tensor): 1 x 3 translation vector.
    """
    device = source.device

    # Compute the weighted centroid of each point cloud
    centroid_source = torch.sum(source * weights, dim=0) / torch.sum(weights)
    centroid_target = torch.sum(target * weights, dim=0) / torch.sum(weights)

    # Subtract the centroids from the point clouds
    source_centered = source - centroid_source
    target_centered = target - centroid_target

    # Compute the covariance matrix between the two point clouds using the weighted correspondences
    covariance_matrix = source_centered.t() @ torch.diag_embed(weights.flatten()) @ target_centered

    # Compute the Singular Value Decomposition (SVD) of the covariance matrix
    U, _, Vt = torch.linalg.svd(covariance_matrix)

    # Compute the rotation matrix and translation vector from the SVD
    R = Vt.t() @ U.t()
    t = centroid_target - centroid_source @ R.t()

    return R, t



