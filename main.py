import numpy as np
import torch
import torch.linalg
import plotly.graph_objs as go
from scipy.spatial.transform import Rotation
import utils
from ICP import ICP
from GNC_GM import GNC_GM
from GNC_TLS import GNC_TLS
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)

noise_levels = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

num_noise_levels = len(noise_levels)

num_runs = 200

num_algorithms = 3

error_data = np.zeros((len(noise_levels), num_algorithms, num_runs))

trans_error_data = np.zeros((len(noise_levels), num_algorithms, num_runs))

SVD_iteration_data = np.zeros((len(noise_levels), num_algorithms, num_runs))


for i, noise_percentage in enumerate(noise_levels):
    noise_level_data_combined = []

    for k in range(num_runs):
        pre_trans_obj = utils.generate_pseudo_bunny(100)

        true_T = utils.generate_rand_transform(distance=1.0)

        # apply transform
        post_trans_obj = (true_T[:3, :3]@pre_trans_obj.T + np.expand_dims(true_T[:3, 3], axis=1)).T

        Y = post_trans_obj

        inlier_mask, Y = utils.apply_outlier_noise(Y, utils.generate_spherical_noise, percentage=noise_percentage, magnitude=5.0)

        # add gaussian noise to all points
        Y = Y + np.random.normal(loc=0, scale=0.01, size=Y.shape)
        Y = torch.tensor(Y)

        X = torch.tensor(pre_trans_obj)

        # plot the original shape and the noisy post transform shape with inliers and outliers marked
        """utils.plot_shapes([pre_trans_obj, Y[inlier_mask == 1], Y[inlier_mask == 0]],
                          colorscale=['#0074D9', '#2ECC40', '#FFDC00'],
                          legends=["original object", "inliers", "outliers"],
                          title=f"Default noise {50}%")
        """

        # Perform baseline Iterative Closest Point
        ICP_transform, ICP_data_labels, ICP_data = ICP(X, Y, inlier_mask, true_T)

        # Perform transform
        ICP_obj = (ICP_transform[:3, :3]@pre_trans_obj.T + np.expand_dims(ICP_transform[:3, 3], axis=1)).T

        # Perform baseline Iterative Closest Point
        GM_transform, GM_data_labels, GM_data = GNC_GM(X, Y, inlier_mask, true_T)

        # Perform transform
        GM_obj = (GM_transform[:3, :3]@pre_trans_obj.T + np.expand_dims(GM_transform[:3, 3], axis=1)).T

        # Perform baseline Iterative Closest Point
        TLS_transform, TLS_data_labels, TLS_data = GNC_TLS(X, Y, inlier_mask, true_T)

        # Perform transform
        TLS_obj = (TLS_transform[:3, :3]@pre_trans_obj.T + np.expand_dims(TLS_transform[:3, 3], axis=1)).T

        # record error data
        error_data[i, 0, k] = ICP_data[-1][0].numpy()
        error_data[i, 1, k] = GM_data[-1][0].numpy()
        error_data[i, 2, k] = TLS_data[-1][0].numpy()

        trans_error_data[i, 0, k] = ICP_data[-1][-1].numpy()
        trans_error_data[i, 1, k] = GM_data[-1][-1].numpy()
        trans_error_data[i, 2, k] = TLS_data[-1][-1].numpy()

        SVD_iteration_data[i, 0, k] = ICP_data[-1][1]
        SVD_iteration_data[i, 1, k] = GM_data[-1][1]
        SVD_iteration_data[i, 2, k] = TLS_data[-1][1]


# generate colors and categories:
categories = []
colors = []
for i in range(num_noise_levels):
    categories.append(noise_levels[i])
    categories.append('')
    categories.append('')
    colors.append('red')
    colors.append('blue')
    colors.append('green')

reshaped_data = np.transpose(error_data, (1, 2, 0))

reshaped_trans_error_data = np.transpose(trans_error_data, (1, 2, 0))

data1 = reshaped_trans_error_data[0, :, :]
data2 = reshaped_trans_error_data[1, :, :]
data3 = reshaped_trans_error_data[2, :, :]

fig, axs = plt.subplots(1, 3, sharey=True)


# plot the first boxplot
box1 = axs[0].boxplot(data1, patch_artist=True)
# set the color of the box and whiskers
for patch in box1['boxes']:
    patch.set_facecolor('blue')

# plot the second boxplot
box2 = axs[1].boxplot(data2, patch_artist=True)
# set the color of the box and whiskers
for patch in box2['boxes']:
    patch.set_facecolor('red')

# plot the third boxplot
box3 = axs[2].boxplot(data3, patch_artist=True)
# set the color of the box and whiskers
for patch in box3['boxes']:
    patch.set_facecolor('green')

xticks = np.array(noise_levels)

for i, ax in enumerate(axs.flatten()):
    ax.set_ylim(0, 1.0)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'],
                  rotation=90)

axs[0].set_ylabel('MSE')
axs[1].set_xlabel('Outlier Noise Level')

axs[0].set_title('ICP (baseline)')
axs[1].set_title('GNC-GM')
axs[2].set_title('GNC-TLS')

fig.suptitle('Mean-Squared-Error of Transform Matrix', fontsize=12)

plt.show()

reshaped_iteration_data = np.transpose(SVD_iteration_data, (1, 2, 0))

iter_data1 = reshaped_iteration_data[0, :, :]
iter_data2 = reshaped_iteration_data[1, :, :]
iter_data3 = reshaped_iteration_data[2, :, :]

fig, axs = plt.subplots(1, 3, sharey=True)


# plot the first boxplot
box1 = axs[0].boxplot(iter_data1, patch_artist=True)
# set the color of the box and whiskers
for patch in box1['boxes']:
    patch.set_facecolor('blue')

# plot the second boxplot
box2 = axs[1].boxplot(iter_data2, patch_artist=True)
# set the color of the box and whiskers
for patch in box2['boxes']:
    patch.set_facecolor('red')

# plot the third boxplot
box3 = axs[2].boxplot(iter_data3, patch_artist=True)
# set the color of the box and whiskers
for patch in box3['boxes']:
    patch.set_facecolor('green')

xticks = np.array(noise_levels)

for i, ax in enumerate(axs.flatten()):
    ax.set_ylim(0, 100)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'],
                  rotation=90)

axs[0].set_ylabel('# Iterations')
axs[1].set_xlabel('Outlier Noise Level')

axs[0].set_title('ICP (baseline)')
axs[1].set_title('GNC-GM')
axs[2].set_title('GNC-TLS')

fig.suptitle('Number of Iterations', fontsize=12)

plt.show()