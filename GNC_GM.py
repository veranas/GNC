import torch
import utils


def GNC_GM(source_points, target_points, inlier_mask, true_T, c_bar=0.1, max_outer_iterations=50, max_inner_iterations=10,
           tolerance=1e-6):
    """
    Graduated Non Convexity-Geman McClure algorithm for point cloud registration.

    Args:
        source_points (torch.Tensor): the source point cloud to be aligned
        target_points (torch.Tensor): the target point cloud
        inlier_mask (list or array-like): boolean mask indicating which points are inliers
        c_bar (float): tuning parameter, should be the maximum expected error of inliers
        max_outer_iterations (int): maximum number of outer iterations before stopping (default=50)
        max_inner_iterations (int): maximum number of inner iterations before stopping (default=10)
        tolerance (float): tolerance level for convergence (default=1e-6)

    Returns:
        cumulative_transformation (torch.Tensor): cumulative transformation to align the source to the target
        data_labels (list): labels for the data log
        data (list): data log with error inliers, number of SVD calculations and weight calculations, and cumulative
                      transformation
    """
    true_T = torch.Tensor(true_T)
    MSE_err = torch.nn.MSELoss()


    # Convert inlier mask to tensor
    y_inliers = torch.tensor(inlier_mask)

    # Convert source and target point clouds to float tensors
    source_points = source_points.float()
    target_points = target_points.float()

    # Initialize the cumulative transformation to the identity matrix and zero vector
    cumulative_R = torch.eye(3)
    cumulative_t = torch.mean(target_points - source_points, axis=0)

    # Initialize the cumulative transformation matrix
    cumulative_transformation = torch.zeros((3, 4))
    cumulative_transformation[:3, :3] = cumulative_R
    cumulative_transformation[:3, 3] = cumulative_t

    # Initialize data logging
    data_labels = ['error_inlier', 'total_SVD_calcs', 'total_weight_calcs', 'mu', 'cumulative_transformation',
                   'transform_error']
    data = []

    # Compute error in inliers for tracking purposes
    error_inlier = utils.MSE_residuals(cumulative_transformation, (source_points * y_inliers.unsqueeze(-1)),
                                       (target_points * y_inliers.unsqueeze(-1)))
    error_inlier = torch.mean(error_inlier)

    # Log data
    data.append([error_inlier, 0, 0, 0.0, cumulative_transformation.clone(),
                 MSE_err(true_T, cumulative_transformation)])

    # Initialize weights at all 1's
    w = torch.ones(source_points.shape[0]).unsqueeze(-1)
    total_weight_calcs = 0

    # Initialize the current source point cloud to be the same as the original source point cloud
    current_source = source_points.clone()
    current_source = current_source + (torch.mean(target_points - source_points))

    # Compute the transformation to align the current source to the target point cloud
    R, t = utils.icp_transformation(current_source, target_points, w)
    total_SVD_calcs = 1

    # Apply the transformation to the current source point cloud
    current_source = current_source @ R.t() + t

    # Update the cumulative transformation
    cumulative_R = R @ cumulative_R
    cumulative_t = R @ cumulative_t + t

    # Calculate error in inliers for tracking purposes
    cumulative_transformation[:3, :3] = cumulative_R
    cumulative_transformation[:3, 3] = cumulative_t

    error_inlier = utils.MSE_residuals(cumulative_transformation, (source_points * y_inliers.unsqueeze(-1)),
                                       (target_points * y_inliers.unsqueeze(-1)))
    error_inlier = torch.mean(error_inlier)

    # Log data
    data.append([error_inlier, total_SVD_calcs, total_weight_calcs, 1.0, cumulative_transformation.clone(), MSE_err(true_T, cumulative_transformation)])

    rmax = torch.max(utils.MSE_residuals(cumulative_transformation, source_points, target_points))
    # initialize mu with the r^2 value

    ###### FOR GNC-GM ##########
    mu_init = 2 * rmax / (c_bar ** 2)

    mu = mu_init

    # Run the optimization loop
    for i in range(max_outer_iterations):

        if i > 0:
            mu /= 1.4
            if mu < 1.0:
                mu = 1.0

        # update weights
        w = utils.update_MSE_weights_GM(cumulative_transformation, w, c_bar, mu, source_points, target_points)
        w = w.reshape(-1, 1)

        total_weight_calcs += 1  # count number of weight calculations

        # Perform ICP iteratively until convergence or until the maximum number of iterations is reached
        for j in range(max_inner_iterations):

            # Compute the transformation to align the current source to the target point cloud
            R, t = utils.icp_transformation(current_source, target_points, w)
            total_SVD_calcs += 1

            # Apply the transformation to the current source point cloud
            current_source = current_source @ R.t() + t

            # Update the cumulative transformation
            cumulative_R = R @ cumulative_R
            cumulative_t = R @ cumulative_t + t

            # calculate the amount of error in inliers for tracking purposes
            cumulative_transformation[:3, :3] = cumulative_R
            cumulative_transformation[:3, 3] = cumulative_t

            error_inlier = utils.MSE_residuals(cumulative_transformation, (source_points * y_inliers.unsqueeze(-1)),
                                         (target_points * y_inliers.unsqueeze(-1)))
            error_inlier = torch.mean(error_inlier)

            data.append([error_inlier, total_SVD_calcs, total_weight_calcs, mu, cumulative_transformation.clone(), MSE_err(true_T, cumulative_transformation)])

            # Check if the transformation has converged
            if torch.all(torch.abs(R - torch.eye(3, device=R.device)) < tolerance) and torch.all(
                    torch.abs(t) < tolerance):
                break

        if mu == 1.0:
            break

    return cumulative_transformation, data_labels, data


