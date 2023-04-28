import numpy as np
import torch
import torch.linalg
import plotly.graph_objs as go
from scipy.spatial.transform import Rotation
import utils
from ICP import ICP
from GNC_GM import GNC_GM
from GNC_TLS import GNC_TLS

pre_trans_obj = utils.generate_pseudo_bunny(1000)

true_T = utils.generate_rand_transform(distance=1.0)

# apply transform
post_trans_obj = (true_T[:3, :3]@pre_trans_obj.T + np.expand_dims(true_T[:3, 3], axis=1)).T

Y = post_trans_obj

inlier_mask, Y = utils.apply_outlier_noise(Y, utils.generate_spherical_noise, percentage=80, magnitude=5.0)

# add gaussian noise to all points
Y = Y + np.random.normal(loc=0, scale=0.01, size=Y.shape)
Y = torch.tensor(Y)

X = torch.tensor(pre_trans_obj)

# plot the original shape and the noisy post transform shape with inliers and outliers marked
"""utils.plot_shapes([pre_trans_obj, X[inlier_mask == 1], X[inlier_mask == 0]],
                  colorscale=['#0074D9', '#2ECC40', '#FFDC00'],
                  legends=["original object", "inliers", "outliers"],
                  title=f"Default noise {50}%")
"""

# Perform baseline Iterative Closest Point
#ICP_transform, ICP_data_labels, ICP_data = ICP(X, Y, inlier_mask)

# Perform transform
#ICP_obj = (ICP_transform[:3, :3]@pre_trans_obj.T + np.expand_dims(ICP_transform[:3, 3], axis=1)).T

# Perform baseline Iterative Closest Point
#GM_transform, GM_data_labels, GM_data = GNC_GM(X, Y, inlier_mask)

# Perform transform
#GM_obj = (GM_transform[:3, :3]@pre_trans_obj.T + np.expand_dims(GM_transform[:3, 3], axis=1)).T

# Perform baseline Iterative Closest Point
#TLS_transform, TLS_data_labels, TLS_data = GNC_TLS(X, Y, inlier_mask)

# Perform transform
#TLS_obj = (TLS_transform[:3, :3]@pre_trans_obj.T + np.expand_dims(TLS_transform[:3, 3], axis=1)).T

#print(f"ICP_error:{ICP_data[-1][0]}, GM_error:{GM_data[-1][0]}, TLS_error:{TLS_data[-1][0]}")

