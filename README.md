# GNC

# Basic Info
This is a simple implementation of Graduated Non-Convexity for point cloud registration running with Iterative Closest Point as the minimizer. This is an algorithm that weights point correspondences to make them progressively more convex or less convex depending on the algorithm used. This algorithm is useful for when trying to perform point cloud registration with a dataset that has a large number of outliers and relatively few datapoints (it can handle up to roughly 80% of the dataset being outliers for some types of evenly dispersed outlier noise, and typically performs well for at least N=100 data points, but may perform worse for higher amounts of data points).

GNC-GM (Geman McClure) will start by weighting the function to be highly convex and gradually revert back the original weights, generally leading towards better convergence towards the global minimum.

GNC-TLS (Truncated Least Squared) will start by weighting the function to be highly convex and gradually revert back the original weights, generally leading towards better convergence towards the global minimum.

ICP (Iterative Closest Point) uses an unweighted version of the Iterative Closest Point algorithm.

# Getting Started
Running main.py will run a demo with 200 iterations of each of the 3 algorithms present, with a custom randomly generated dataset.

Running samples.py will create a series of 3D plotly plots that demonstrate the results of the 3 algorithms in action
