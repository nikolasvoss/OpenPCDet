import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from numba import jit

feature_maps = None


def extractFeatureMapHook(module, input, output):
    """Extracts the output of the layer for visualization."""
    global feature_maps
    # when feature_maps is not empty, add output to feature_maps as a list
    if feature_maps is not None:
        feature_maps.append(output)
    else:
        feature_maps = [output]


def registerHookForLayer(model, layer_names):
    """Registers hooks for layers in the model.

    Args:
    model: The neural network model to which the hooks will be attached.
    layer_names: A string or a list of strings with the unique names of the
                 layers within the model's scope.

    Raises:
    TypeError: If `layer_names` is neither a string nor a list of strings.
    ValueError: If any specified `layer_name` does not correspond to a layer in the model."""

    # Ensure layer_names is a list
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    elif not isinstance(layer_names, list) or not all(isinstance(name, str) for name in layer_names):
        raise TypeError('layer_names must be a string or a list of strings.')

    # Loop over all layer names to register hooks
    for layer_name in layer_names:
        layer_exists = False
        for name, module in model.named_modules():
            if name == layer_name:
                # Register a forward hook on the layer to capture feature maps
                module.register_forward_hook(extractFeatureMapHook)
                layer_exists = True
                break

        if not layer_exists:
            # Throw an exception if the layer is not found
            raise ValueError(f'Layer {layer_name} not found in the model.')


def printAllModelLayers(model):
    """Prints all layers of the model."""
    for name, module in model.named_modules():
        print(name)


def vis_fmap_2d(feature_map, output_dir, batch_idx=0, fmap_indices=None, z_plane_indices=None, no_negative_values=False, layer_name="", visibility_factor=1.4):
    """Visualizes feature maps as slices in the z-plane using matplotlib.
    The plots are saved as images in the specified output directory.

    Args:
    feature_map [batch_size, fmaps, z, y, x], [batch_size, fmaps, y, x]: The feature map tensor to visualize.
    output_dir (str): The directory to save the visualization images.
    batch_idx (int, [int]): The index of the batch to visualize. Default is 0. Currently only supports one batch at a time.
    fmap_indices (int, [int]): The index of the feature map to visualize. Default is None, which means all are being visualized.
    z_plane_indices (int, [int]): The index of the z-plane to visualize. Default is None, which means all are being visualized.
    no_negative_values (bool): If True, the feature map values are set to their absolute values. Default is False.

    Raises:
    ValueError: If the feature_map is None or data types are incorrect or indices are out of bound.
    FileNotFoundError: If the output directory does not exist.
    """
    if feature_map is None:
        raise ValueError("No feature map available. Check if the hook was triggered correctly.")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory `{output_dir}` does not exist.")

    if hasattr(feature_map, 'dense'):
        feature_map = feature_map.dense()
    feature_map = feature_map.detach()

    # Prepare to visualize only specified feature maps or z-planes (if it exists)
    num_feature_maps = feature_map.size(1)
    if feature_map.ndim == 4:
        num_z_planes = 1
        z_plane_indices = 0
        feature_map = feature_map.unsqueeze(2)  # Add a dummy z-dimension for compatibility
        print("Feature map is 2D. Only one z-plane available. Setting z_plane to 0.")
    else: # 3D feature map
        num_z_planes = feature_map.size(2)
        print(f"Feature map is 3D. Number of z-planes: {num_z_planes}")

    # If no feature map indices are provided, visualize all feature maps
    if fmap_indices is None:
        fmap_indices = list(range(num_feature_maps))

    # check file types
    if not isSingleIntOrListOfInts(batch_idx):
        raise ValueError("batch_idx must be an integer or a list of integers.")
    if not isSingleIntOrListOfInts(fmap_indices):
        raise ValueError("fmap_indices must be an integer or a list of integers.")
    if not isSingleIntOrListOfInts(z_plane_indices):
        raise ValueError("z_plane_indices must be an integer or a list of integers.")

    # turn passed integers into lists to be iterable
    if isinstance(batch_idx, int):
        batch_idx = [batch_idx]
    if isinstance(fmap_indices, int):
        fmap_indices = [fmap_indices]
    if isinstance(z_plane_indices, int):
        z_plane_indices = [z_plane_indices]

    # Check if passed indices are within bounds
    for idx in fmap_indices:
        if not (0 <= idx < num_feature_maps):
            raise ValueError(f"fmap_idx {idx} is out of bounds. It must be between 0 and {num_feature_maps - 1}")

    for idx in batch_idx:
        if not (0 <= idx < feature_map.size(0)):
            raise ValueError(f"batch_idx {idx} is out of bounds. It must be between 0 and {feature_map.size(0) - 1}")

    for idx in z_plane_indices:
        if not (0 <= idx < num_z_planes):
            raise ValueError(f"z_plane_idx {idx} is out of bounds. It must be between 0 and {num_z_planes - 1}")

    # Visualization loop
    for fmap_idx in fmap_indices:
        for z_plane in z_plane_indices:
            # Extract the specific slice to visualize
            # squeeze(0) removes the z-dimension if it is 1
            feature_slice = feature_map[batch_idx, fmap_idx, z_plane].squeeze(0).cpu().numpy()
            # flip y axis
            feature_slice = np.flipud(feature_slice)
            plt.title(f'Batch {batch_idx[0]}, Feature Map {fmap_idx}, Z-Plane {z_plane}')
            plt.figure(figsize=(8, 8))  # Set the figure size to be 8x8 inches
            if no_negative_values:
                feature_slice = feature_slice[:, :] - feature_slice[15, 15] # hack to set empty space to 0
                plt.imshow(abs(feature_slice*visibility_factor), cmap='copper', vmin=0, vmax=abs(feature_slice).max())
            else:
                limit = np.min([abs(feature_slice.min()), abs(feature_slice.max())])
                plt.imshow(feature_slice*visibility_factor, cmap='PRGn', vmin=-limit, vmax=limit)
            plt.colorbar(fraction=0.0453)
            # add title
            plt.title(f'Layer: {layer_name}')
            file_name = os.path.join(output_dir, (f'map_batch{batch_idx[0]}_fmap{fmap_idx}_z{z_plane}.jpg'))
            print(f"Saving image to {file_name}")
            plt.savefig(os.path.join(output_dir, file_name), dpi=500) # > 1024x1024 pixels (8 inches * 128 DPI)
            plt.close()


def vis_fmap_3d(feature_map, output_dir, samples_idx, batch_idx=0, fmap_indices=None, input_points=None, same_plot=False, gt_boxes=None, pred_boxes=None):
    """Visualizes 3D feature maps using Open3D.

    Args:
    feature_map [batch_size, feature_maps, z, y, x]: The feature map tensor to visualize.
    output_dir (str): The directory to save the visualization images.
    batch_idx (int): The index of the batch to visualize. Default is 0.
    fmap_indices (int): The index of the feature map to visualize. Default is None, which means all are being visualized.

    Raises:
    ValueError: If the feature_map is None or the output_dir does not exist.
    """
    if feature_map is None:
        raise ValueError("No feature map available. Check if the hook was triggered correctly.")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory `{output_dir}` does not exist.")
    # check file types
    if not isSingleIntOrListOfInts(batch_idx):
        raise ValueError("batch_idx must be an integer or a list of integers.")
    if not isSingleIntOrListOfInts(fmap_indices):
        raise ValueError("fmap_indices must be an integer or a list of integers.")

    # File operations
    if hasattr(feature_map, 'dense'):
        feature_map = feature_map.dense()
    feature_map = feature_map.detach()
    if feature_map.ndim == 4:
        print('2D feature map detected')
        feature_map = feature_map.unsqueeze(2)  # Add a dummy z-dimension for compatibility
    num_feature_maps = feature_map.shape[1]

    # turn passed integers into lists to be iterable
    if isinstance(batch_idx, int):
        batch_idx = [batch_idx]
    if isinstance(fmap_indices, int):
        fmap_indices = [fmap_indices]
    # If no feature map indices are provided, visualize all feature maps
    if fmap_indices is None:
        fmap_indices = list(range(num_feature_maps))
    # Check if passed indices are within bounds
    for idx in fmap_indices:
        if not (0 <= idx < num_feature_maps):
            raise ValueError(f"fmap_indices {idx} is out of bounds. It must be between 0 and {num_feature_maps - 1}")
    for idx in batch_idx:
        if not (0 <= idx < feature_map.size(0)):
            raise ValueError(f"batch_idx {idx} is out of bounds. It must be between 0 and {feature_map.size(0) - 1}")
    if input_points is not None:
        input_points = input_points[:, 1:4]  # Only use the xyz coordinates

    for z_idx in range(1):
        for fmap_idx in fmap_indices:
            z_idx=5
            # one plot for each feature map

            # squeeze removes leftover dimensions if they are 1
            single_feature_map = feature_map[batch_idx, fmap_idx, z_idx].squeeze().unsqueeze(0) # values[z,y,x] / [1,y,x], remove features dimension
            z, y, x = torch.nonzero(single_feature_map, as_tuple=True)
            # scale values for colors
            nonzero_values = single_feature_map[z.cpu(), y.cpu(), x.cpu()]

            # create a nparray of size 3 x N with nonzero_values in the first row, the rest is 0
            colors = np.zeros((3, len(nonzero_values)))
            colors[0] = nonzero_values.cpu().numpy()
            colors = colors.astype(np.float64)  # Convert to float64, as Open3D expects

            # Point cloud range from nuscenes.yaml
            pointcloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            # original voxel size from nuscenes.yaml
            voxel_size = [0.1, 0.1, 0.2]
            # voxel size fitted to current feature map
            voxel_size[0] = (pointcloud_range[3] - pointcloud_range[0]) / single_feature_map.shape[2] # x
            voxel_size[1] = (pointcloud_range[4] - pointcloud_range[1]) / single_feature_map.shape[1] # y
            voxel_size[2] = (pointcloud_range[5] - pointcloud_range[2]) / single_feature_map.shape[0] # z
            del single_feature_map

            x_meters = x.cpu().numpy() * voxel_size[0] - 51.2  # voxel size and pc range
            y_meters = y.cpu().numpy() * voxel_size[1] - 51.2
            z_meters = z.cpu().numpy() * voxel_size[2] - 4.9

            points = o3d.utility.Vector3dVector(np.vstack((x_meters, y_meters, z_meters)).T)
            colors = o3d.utility.Vector3dVector(colors.T)
            feature_pcd = o3d.geometry.PointCloud()
            feature_pcd.points = points
            feature_pcd.colors = colors

            # WARNING: the voxels z-dimension is currently not correct
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(feature_pcd, voxel_size=voxel_size[0])

            # Create Open3d Visualizer:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f'Voxel Size: {voxel_size}')
            vis.get_render_option().point_size = 2.5
            vis.get_render_option().background_color = [0.5,0.5,0.5]

            # Input Points Visualization
            if input_points is not None and same_plot:
                input_point_cloud = o3d.geometry.PointCloud()
                input_point_cloud.points = o3d.utility.Vector3dVector(input_points.detach().cpu().numpy().astype(np.float64))
                input_point_cloud.paint_uniform_color([0, 0.8, 0])  # green
                vis.add_geometry(input_point_cloud)
                vis.add_geometry(voxel_grid)

                if pred_boxes is not None:
                    vis, box3d_list = drawPredBoxes(vis, pred_boxes)
                if gt_boxes is not None:
                    gt_angles = gt_boxes[:, 6:8].reshape((-1, 2))
                    vis, box3d_list = drawGtBoxes(vis, gt_boxes)
            elif input_points is not None and not same_plot:
                vis.add_geometry(voxel_grid)

                input_point_cloud = o3d.geometry.PointCloud()
                input_point_cloud.points = o3d.utility.Vector3dVector(input_points.detach().cpu().numpy().astype(np.float64))
                input_point_cloud.paint_uniform_color([0, 1, 0])  # green
                vis.create_window()
                vis.add_geometry(input_point_cloud)
            else:  # no input points were passed
                vis.add_geometry(voxel_grid)
            vis.get_view_control().set_zoom(0.3)
            # change field of view
            vis.get_view_control().change_field_of_view(step=-90)
            vis.run()
            # save the entropy image.
            if output_dir is not None:
                print(f"Saving image to {output_dir}fmap_sample{samples_idx}_channel{fmap_idx}_z{z_idx}.png")
                vis.capture_screen_image(
                    f'{output_dir}fmap_sample{samples_idx}_channel{fmap_idx}_z{z_idx}.png')

            vis.destroy_window()


def vis_fmap_entropy_3d(feature_map, samples_idx, output_dir=None, input_points=None, pred_boxes=None, gt_boxes=None, z_plane_idx=0):
    if feature_map is None:
        raise ValueError("No feature map available. Check if the hook was triggered correctly.")
    if output_dir is not None and not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory `{output_dir}` does not exist.")
    if input_points is not None:
        input_points = input_points[:, 1:4]  # Only use the xyz coordinates

    if hasattr(feature_map, 'dense'):
        feature_map = feature_map.dense()
    feature_map = feature_map.detach()
    if feature_map.ndim == 4:
        print('2D feature map detected')
        feature_map = feature_map.unsqueeze(2) # Add a dummy z-dimension for compatibility
    # feature_map should have the shape [batch_size, feature_maps, z, y, x]
    # Calculate entropy over all channels for every single spatial location ("voxel")
    fmap_entropy, num_bins = calc_fmap_entropy(feature_map[:,:,z_plane_idx].cpu().numpy().squeeze(), num_bins=15)
    fmap_entropy = fmap_entropy - fmap_entropy.min()
    fmap_entropy = fmap_entropy / fmap_entropy.max()
    fmap_entropy[fmap_entropy < 0.05] = 0

    ############################################################
    # Open3D Visualization
    ############################################################
    # Point cloud range from nuscenes.yaml
    pointcloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    # original voxel size from nuscenes.yaml
    voxel_size = [0.1, 0.1, 0.2]
    # voxel size fitted to current feature map
    voxel_size[0] = (pointcloud_range[3] - pointcloud_range[0]) / feature_map.shape[-1]  # x
    voxel_size[1] = (pointcloud_range[4] - pointcloud_range[1]) / feature_map.shape[-2]  # y
    voxel_size[2] = (pointcloud_range[5] - pointcloud_range[2]) / feature_map.shape[-3]  # z
    print('Voxel size: ', voxel_size)

    x = np.linspace(pointcloud_range[0], pointcloud_range[3], feature_map.shape[4], endpoint=False)
    y = np.linspace(pointcloud_range[1], pointcloud_range[4], feature_map.shape[3], endpoint=False) # added minus, because the y-axis was flipped
    z = np.array([-5.])

    # Create Open3d Visualizer:
    points = npVectorToO3dPoints(x, y, z)
    colors = np.zeros((len(fmap_entropy.flatten()), 3), dtype=np.float64)
    colors[:, 0] = fmap_entropy.flatten()
    colors = o3d.utility.Vector3dVector(colors)

    feature_pcd = o3d.geometry.PointCloud()
    feature_pcd.points = points
    feature_pcd.colors = colors

    # # WARNING: the voxels z-dimension is currently not correct
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(feature_pcd, voxel_size=voxel_size[0])

    # Create Open3d Visualizer:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f'Sample: {samples_idx}, Voxel Size: {voxel_size}, num_bins={num_bins}')
    vis.get_render_option().point_size = 2.5
    vis.get_render_option().background_color = [1,1,1]

    # plot the sample pc
    # input_point_cloud = o3d.geometry.PointCloud()
    # input_point_cloud.points = o3d.utility.Vector3dVector(input_points.detach().cpu().numpy().astype(np.float64))
    # input_point_cloud.paint_uniform_color([0, 1, 0])  # green
    # vis.add_geometry(input_point_cloud)

    # plot gt and pred boxes
    if pred_boxes is not None:
        vis, box3d_list = drawPredBoxes(vis, pred_boxes)
    if gt_boxes is not None:
        # gt_angles = gt_boxes[:, 6:8].reshape((-1, 2))
        vis, box3d_list = drawGtBoxes(vis, gt_boxes)
    
    vis.add_geometry(voxel_grid)

    vis.get_view_control().set_zoom(0.3)
    # change field of view
    vis.get_view_control().change_field_of_view(step=-90)
    vis.run()
    # save the entropy image. add x_shift, multiplier and num_bins to the filename
    if output_dir is not None:
        print(f"Saving image to {output_dir}entropy_sample{samples_idx}_numbins{num_bins}.png")
        vis.capture_screen_image(f'{output_dir}entropy_sample{samples_idx}_numbins{num_bins}.png')

    vis.destroy_window()


def calc_fmap_entropy(feature_map, num_bins=None):
    """
        Compute the normalized entropy of a feature map tensor.

        This function takes a feature map tensor with dimensions [feature_maps, y, x]
        and calculates its entropy after applying outlier removal, normalization, and
        equal-width binning for probability estimation.

        Parameters:
        feature_map (ndarray): A numpy array representing the feature map tensor with shape
                               [feature_maps, y, x].

        Returns:
        ndarray: A 2D numpy array representing the normalized entropy over all feature maps
                 for each spatial location [y, x].
        int: The number of bins used for histogram binning.

        Notes:
        - The function computes the entropy over feature maps, and it performs several pre-processing
          steps including (i) removal of outliers based on percentile, (ii) normalization of values
          to the range [0, 1], and (iii) probability estimation using histogram binning.
        - The output entropy is normalized to the range [0, 1].
        - The function handles the case of an empty feature map by returning zeros.
        - An epsilon value 'e' is added to probabilities to avoid logarithm of zero errors.
        - Binning of feature map values is done using a number of bins dependent on the number of channels
         and places them between 0 and 1.
        """

    # Expects a feature map tensor of shape [feature_maps, y, x]
    # Abort if the feature map is empty , return shape is [y, x]
    if feature_map.max() == 0 and feature_map.min() == 0:
        return np.zeros([feature_map.shape[-2], feature_map.shape[-1]], device=feature_map.device), 0
    lower_limit = np.percentile(feature_map[feature_map != 0], 0.01)
    upper_limit = np.percentile(feature_map[feature_map != 0], 99.99)
    # Remove outliers to get better histogram
    feature_map[feature_map < lower_limit] = lower_limit
    feature_map[feature_map > upper_limit] = upper_limit
    # Normalize for visualization
    feature_map -= lower_limit
    feature_map = feature_map * 1 / (upper_limit-lower_limit)

    # Number of bins , fewer often work better for visualization
    if num_bins is None:
        num_bins = max(3, feature_map.shape[0] // 20) # Arbitrary value for visualization
    else:
        assert isinstance(num_bins, int), "num_bins must be an integer."
        assert num_bins > 0, "num_bins must be greater than 0."
    # Create histograms
    bin_edges = np.linspace(0, 1, num_bins + 1)
    histograms = computeHistograms(feature_map, bin_edges, num_bins)
    # Add small constants to avoid probabilities = 0 and division by zero.
    rel_frequency = (histograms / (histograms.sum(axis=0, keepdims=True) + 1e-10)).clip(min=1e-10)
    # Calculate entropy over channel dimension
    entropy = -np.sum(rel_frequency * np.log(rel_frequency), axis=0)
    return entropy, num_bins


# not used for thesis, not sure if it works
def calc_fmap_entropy_dense(feature_map, num_bins=None):
    """
    Compute the normalized entropy of a feature map tensor.

    This function takes a feature map tensor with dimensions [batch_size, feature_maps, y, x]
    and calculates its entropy after applying outlier removal, normalization, and
    equal-width binning for probability estimation.

    Parameters:
    feature_map (torch.Tensor): A torch tensor representing the feature map tensor with shape
                                [batch_size, feature_maps, y, x].

    Returns:
    torch.Tensor: A 2D torch tensor representing the normalized entropy over all feature maps
                  for each spatial location [batch_size, y, x].
    int: The number of bins used for histogram binning.
    """

    # Check if feature_map is a tensor
    if not torch.is_tensor(feature_map):
        raise ValueError("feature_map must be a torch tensor.")

    # Expects a feature map tensor of shape [feature_maps, y, x]
    # Abort if the feature map is empty
    if feature_map.max() == 0 and feature_map.min() == 0:
        # return shape is [batch size, y, x]
        return torch.zeros([feature_map.shape[0], feature_map.shape[-2], feature_map.shape[-1]], device=feature_map.device), 0

    feature_map = feature_map.to(torch.float32)
    # Remove outliers to get usable histogram
    lower_limit = torch.quantile(feature_map[feature_map != 0], 0.01)
    upper_limit = torch.quantile(feature_map[feature_map != 0], 0.99)
    # set all values below or above limits to limits
    feature_map = torch.clamp(feature_map, min=lower_limit, max=upper_limit)

    # normalize
    feature_map -= lower_limit
    feature_map = torch.div(feature_map, (upper_limit - lower_limit))  # set maximum value to 1

    # num_bins, fewer often work better
    if num_bins is None:
        num_bins = max(3, feature_map.shape[0] // 20)
    else:
        assert isinstance(num_bins, int), "num_bins must be an integer."
        assert num_bins > 0, "num_bins must be greater than 0."

    # Compute histograms
    bin_edges = torch.linspace(0, 1, num_bins + 1)
    histograms = computeHistogramsTorch(feature_map, bin_edges, num_bins).float()
    # add a small constant to avoid division by zero
    # add a small constant to avoid probabilities = 0
    probabilities = torch.div(histograms, (histograms.sum(dim=1, keepdim=True) + 1e-10)).clamp(min=1e-10)

    entropy = -torch.sum(probabilities * torch.log(probabilities), dim=1)

    return entropy, num_bins


def calc_fmap_entropy_sparse(feature_map, num_bins=None):
    """
    does the same as calc_fmap_entropy_dense, but for sparse tensors
    indices are in feature_map.indices, values in feature_map
    batch index is in feature_map.indices[:, 0]

    input: feature_map: SparseConvTensor of shape [points, feature_maps]

    returns:
    feature_map of shape [points, 1]

    """
    # Check if feature_map is a tensor
    if not torch.is_tensor(feature_map):
        raise ValueError("feature_map must be a torch tensor.")

    feature_map = feature_map.to(torch.float32) # somehow does not work with float16. Could be issue with  old torch version.
    # Cut outliers and normalize. Remove zeros first
    lower_limit = torch.quantile(feature_map[feature_map != 0], 0.01)
    upper_limit = torch.quantile(feature_map[feature_map != 0], 0.99)

    # set all values below or above limits to limits
    feature_map = torch.clamp(feature_map, min=lower_limit, max=upper_limit)

    # normalize
    feature_map -= lower_limit
    feature_map = torch.div(feature_map, (upper_limit - lower_limit))  # set maximum value to 1

    # num_bins, fewer often work better
    if num_bins is None:
        num_bins = max(3, feature_map.shape[1] // 20)
    else:
        assert isinstance(num_bins, int), "num_bins must be an integer."
        assert num_bins > 0, "num_bins must be greater than 0."

    # Compute histograms
    bin_edges = torch.linspace(0, 1, num_bins + 1)
    histograms = torch.zeros((feature_map.shape[0], num_bins), dtype=torch.float32, device=feature_map.device)
    for bin in range(num_bins):
        histograms[:, bin] = torch.sum(
            (feature_map >= bin_edges[bin]) &
            (feature_map < bin_edges[bin+1]),
            dim=1)

    # add a small constant to avoid division by zero
    # add a small constant to avoid probabilities = 0
    probabilities = torch.div(histograms, (histograms.sum(dim=1, keepdim=True) + 1e-10)).clamp(min=1e-10)

    entropy = -torch.sum(probabilities * torch.log(probabilities), dim=1)

    return entropy


@jit(nopython=True)
def computeHistograms(feature_map, bin_edges, num_bins):
    # Iterate over all voxels and compute the histogram for each. Use for visualization.
    h, w = feature_map.shape[1], feature_map.shape[2]
    histograms = np.zeros((num_bins, h, w), dtype=np.uint32)
    for i in range(h):
        for j in range(w):
            histograms[:, i, j], _ = np.histogram(feature_map[:, i, j], bins=bin_edges)
    return histograms


def computeHistogramsTorch(feature_map, bin_edges, num_bins):
    # find all values that are in the bins bin_edges[n] <= x < bin_edges[n+1] and count them.
    # expects: feature_map of shape [batch, channels, y, x]
    batch, h, w = feature_map.shape[0], feature_map.shape[-2], feature_map.shape[-1]
    histograms = torch.zeros((batch, num_bins, h, w), dtype=torch.int32, device=feature_map.device)
    for bat in range(feature_map.shape[0]):
        for bin in range(num_bins):
            histograms[bat, bin] = torch.sum(
                (feature_map[bat] >= bin_edges[bin]) &
                (feature_map[bat] < bin_edges[bin+1]),
                dim=0)
        # Handle the last bin inclusive of the upper edge
        histograms[-1] = (feature_map[bat] >= bin_edges[-2]).sum(dim=0)
    return histograms


def drawPredBoxes(vis, pred_boxes):
    vis, box3d_list = draw_box(vis, pred_boxes.cpu(), (1, 1, 0))
    print('Number of Pred-Boxes: ', pred_boxes.shape[0])
    return vis, box3d_list


def drawGtBoxes(vis, gt_boxes):
    for i in range(gt_boxes.shape[0]):
        gt_box = gt_boxes[i, :].reshape((1, 9))
        vis, box3d_list = draw_box(vis, gt_box.cpu(), (0, 1, 1))
    print('Number of GT-Boxes: ', gt_boxes.shape[0])
    return vis, box3d_list
    
    
def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    box_colormap = [
        [1, 1, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 1, 0],
    ]
    box3d_list = []

    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)
        box3d_list.append(box3d)

    return vis, box3d_list


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    print('rot: ', rot)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d


def isSingleIntOrListOfInts(value):
    # First, check if the value is a single integer
    if isinstance(value, int):
        return True
    # Then, check if it is a list
    elif isinstance(value, list):
        # Check if all elements in the list are integers
        return all(isinstance(item, int) for item in value)
    # If it's neither an integer nor a list of integers, return False
    return False


def sumChannelsPerPixel(feature_map):
    """
    Sum the feature maps over all channels for each pixel.
    Normalize the values to the range [0, 1].

    Parameters:
    feature_map (torch.Tensor or np.ndarray): A tensor or array representing the feature map tensor with shape
                                [batch_size, feature_maps, y, x].

    Returns:
    torch.Tensor or np.ndarray: A 2D tensor or array representing the sum of all feature maps over all channels
                  for each spatial location [batch_size, y, x].
    """
    if isinstance(feature_map, np.ndarray):
        sum_channels = np.sum(feature_map, axis=0, keepdims=False)
        sum_channels -= sum_channels.min()
        sum_channels = np.divide(sum_channels, sum_channels.max())
        return sum_channels
    elif isinstance(feature_map, torch.Tensor):
        sum_channels = torch.sum(feature_map, dim=1, keepdim=False)
        sum_channels -= sum_channels.min()
        sum_channels = torch.div(sum_channels, sum_channels.max())
        return sum_channels


def npVectorToO3dPoints(x:np.array, y:np.array=None, z:np.array=None):
    """Converts numpy arrays to Open3D Vector3dVector.
    If y or z are not provided, they are set to 0.
    Expects:
        x: np.array of shape (N,)
        y: np.array of shape (M,) or None
        z: np.array of shape (K,) or None
    Returns:
        o3d.utility.Vector3dVector of shape (N*M*K, 3)
    usage:
    x = np.linspace(-3, 3, 401)
    y = ...
    pcd = o3d.geometry.PointCloud()
    pcd.points = npVectorToO3dPoints(x, y, z)
    """
    if y is None:
        y = np.zeros_like(x)
    if z is None:
        z = np.zeros_like(x)
    mesh_x, mesh_y, mesh_z = np.meshgrid(x, y, z)
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = mesh_x.flatten()
    xyz[:, 1] = mesh_y.flatten()
    xyz[:, 2] = mesh_z.flatten()

    return o3d.utility.Vector3dVector(xyz)


def plotHistAndBoxplot(fmap_entropy, bins=100):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist(fmap_entropy.flatten(), bins=100, color='blue', alpha=0.7)
    ax[0].set_title('Histogram')
    ax[0].set_ylabel('Frequency')
    ax[1].boxplot(fmap_entropy.flatten(), vert=False, showflyers=True)
    ax[1].set_title('Boxplot')
    plt.tight_layout()
    plt.show()