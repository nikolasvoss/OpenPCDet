import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

global feature_maps

def extractFeatureMapHook(module, input, output):
    """Extracts the output of the layer for visualization."""
    global feature_maps
    feature_maps = output

def registerHookForLayer(model, layer_name):
    """Registers a hook for one layer in the model. Call in loop for multiple layers.

    Args:
    model: The neural network model to which the hook will be attached.
    layer_name (str): The unique name of the layer within the model's scope.

    Raises:
    TypeError: If the `layer_name` is not a string.
    ValueError: If the specified `layer_name` does not correspond to a layer in the model."""
    if not isinstance(layer_name, str):
        raise TypeError(f'layer_name must be a string, got {type(layer_name)} instead. Provide only one layer at a time.')

    layer_exists = False
    for name, module in model.named_modules():
        if name == layer_name:
            # Register a forward hook on the layer to capture feature maps
            module.register_forward_hook(extractFeatureMapHook)
            layer_exists = True
            break

    if not layer_exists:
        # throw exception if the layer is not found
        raise ValueError(f'Layer {layer_name} not found in the model.')

def registerHookForLayers(model, layer_names):
    """
    Registers hooks for multiple specified layers in the model.
    Calls registerHookForLayer in a loop for each layer.

    Args:
    model: The neural network model.
    layer_names: A list of layer names to register hooks on.

    Raises:
    TypeError: If layer_names is not a list.
    """
    if not isinstance(layer_names, list):
        raise TypeError('layer_names must be a list of strings.')

    # Register a hook for each layer in layer_names
    for layer_name in layer_names:
        try:
            registerHookForLayer(model, layer_name)
            print(f"Hook registered for layer: {layer_name}")
        except ValueError as e:
            print(f"Error: {e}")
        except TypeError as e:
            print(f"Error: {e}")


def visualizeFeatureMap(feature_map, output_dir, batch_idx=0, fmap_idx=None, z_plane=None):
    """Visualizes feature maps as slices in the z-plane using matplotlib.
    The plots are saved as images in the specified output directory.

    Args:
    feature_map [batch_size, feature_maps, z, y, x]: The feature map tensor to visualize.
    output_dir (str): The directory to save the visualization images.
    batch_idx (int): The index of the batch to visualize. Default is 0.
    fmap_idx (int): The index of the feature map to visualize. Default is None, which means all are being visualized.
    z_plane (int): The index of the z-plane to visualize. Default is None, which means all are being visualized.

    Raises:
    ValueError: If the feature_map is None or the output_dir does not exist.
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
        print("Feature map is 2D. Only one z-plane available. Setting z_plane to 0.")
        num_z_planes = 1
        z_plane = 0
        feature_map = feature_map.unsqueeze(2)  # Add a dummy z-dimension for compatibility
    else: # 3D feature map
        num_z_planes = feature_map.size(2)
        print(f"Feature map is 3D. Number of z-planes: {num_z_planes}")

    feature_map_indices = [fmap_idx] if fmap_idx is not None else range(num_feature_maps)
    z_plane_indices = [z_plane] if z_plane is not None else range(num_z_planes)

    # Check if fmap_idx is provided as a list, if not and it is not None, make it a list
    if isinstance(fmap_idx, list):
        fmap_indices = fmap_idx
    elif fmap_idx is not None:
        fmap_indices = [fmap_idx]
    else:
        fmap_indices = range(num_feature_maps)
    # Check if passed indices are within bounds
    for idx in fmap_indices:
        if not (0 <= idx < num_feature_maps):
            raise ValueError(f"fmap_idx {idx} is out of bounds. It must be between 0 and {num_feature_maps - 1}")

    if batch_idx < 0 or batch_idx >= feature_map.shape[0]:
        raise ValueError(f"batch_idx is out of bounds. It must be between 0 and {feature_map.shape[0] - 1}")

    visibility_factor = 4  # multiply all values for better visibility

    # Visualization loop
    for fmap_idx in feature_map_indices:
        for z_plane in z_plane_indices:
            # Extract the specific slice to visualize
            # squeeze(0) removes the z-dimension if it is 1
            feature_slice = feature_map[batch_idx, fmap_idx, z_plane].squeeze(0).cpu().numpy()

            plt.figure(figsize=(8, 8))  # Set the figure size to be 8x8 inches
            plt.imshow(feature_slice*visibility_factor, cmap='PRGn', vmax=1, vmin=-1)
            plt.colorbar()
            file_name = os.path.join(output_dir, (f'map_batch{batch_idx}_fmap{fmap_idx}_z{z_plane}.jpg'))
            plt.savefig(os.path.join(output_dir, file_name), dpi=150) # > 1024x1024 pixels (8 inches * 128 DPI)
            plt.close()

def visualizeFeatureMap3d(feature_map, output_dir, batch_idx=0, fmap_idx=None, input_points=None, same_plot=False):
    """Visualizes 3D feature maps using matplotlib.

    Args:
    feature_map [batch_size, feature_maps, z, y, x]: The feature map tensor to visualize.
    output_dir (str): The directory to save the visualization images.
    batch_idx (int): The index of the batch to visualize. Default is 0.
    fmap_idx (int): The index of the feature map to visualize. Default is None, which means all are being visualized.

    Raises:
    ValueError: If the feature_map is None or the output_dir does not exist.
    """
    if feature_map is None:
        raise ValueError("No feature map available. Check if the hook was triggered correctly.")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory `{output_dir}` does not exist.")

    # File operations
    feature_map = feature_map.dense().detach()
    num_feature_maps = feature_map.shape[1]

    # Check if fmap_idx is provided as a list, if not and it is not None, make it a list
    if isinstance(fmap_idx, list):
        fmap_indices = fmap_idx
    elif fmap_idx is not None:
        fmap_indices = [fmap_idx]
    else:
        fmap_indices = range(num_feature_maps)
    # Check if passed indices are within bounds
    for idx in fmap_indices:
        if not (0 <= idx < num_feature_maps):
            raise ValueError(f"fmap_idx {idx} is out of bounds. It must be between 0 and {num_feature_maps - 1}")

    if batch_idx < 0 or batch_idx >= feature_map.shape[0]:
        raise ValueError(f"batch_idx is out of bounds. It must be between 0 and {feature_map.shape[0] - 1}")

    plt.style.use('default')

    if input_points is not None:
        input_points = input_points[:, 1:4]  # Only use the xyz coordinates
        if same_plot:
            fig = plt.figure(figsize=(11, 11))
            ax_feature = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.figure(figsize=(22, 11))
            ax_feature = fig.add_subplot(121, projection='3d')

        # ax_input is added later if same_plot is False
        plt.subplots_adjust(left=0.0, right=0.95, bottom=0.0, top=1.0, wspace=0.0, hspace=0.4)
    else:
        fig = plt.figure(figsize=(11, 11))
        ax_feature = fig.add_subplot(111, projection='3d')

    for fmap_idx in fmap_indices:
        # one plot for each feature map

        ax_feature.w_xaxis.pane.fill = False
        ax_feature.w_yaxis.pane.fill = False
        ax_feature.w_zaxis.pane.fill = False
        # Set labels and title
        ax_feature.set_xlabel('X axis')
        ax_feature.set_ylabel('Y axis')
        ax_feature.set_zlabel('Z axis')
        ax_feature.set_title(f'3D Feature Map - Batch: {batch_idx}, Feature Map: {fmap_idx}')

        single_feature_map = feature_map[batch_idx, fmap_idx] # values[z,y,x]
        z, y, x = torch.nonzero(single_feature_map, as_tuple=True)
        x_meters = x.cpu().numpy() * 0.1 - 51.2  # voxel size and pc range
        y_meters = y.cpu().numpy() * 0.1 - 51.2
        z_meters = z.cpu().numpy() * 0.2 - 5.0
        nonzero_values = single_feature_map[z, y, x].cpu().numpy()

        # Plot using scatter to create a 3D voxel visualization
        img = ax_feature.scatter(x_meters,
                         y_meters,
                         z_meters,
                         c=abs(nonzero_values), # negative values are for loosers
                         cmap='copper',
                         s=1,
                         vmax=5)

        # use when negative values should be visible
        # img = ax_feature.scatter(x.cpu().numpy(),
        #                  y.cpu().numpy(),
        #                  z.cpu().numpy(),
        #                  c=nonzero_values,
        #                  cmap='PRGn',
        #                  marker='.',
        #                  vmax=1,
        #                  vmin=-1)

        # Input Points Visualization
        if input_points is not None and same_plot:
            x, y, z = input_points.cpu().numpy().T
            ax_feature.scatter(x, y, z, c='green', s=1, label='Input Points')  # Plot input points in the same plot
            ax_feature.legend()

        plt.colorbar(img, shrink=0.2, aspect=10, ax=ax_feature)

        if input_points is not None and not same_plot:
            ax_input = fig.add_subplot(122, projection='3d')
            x, y, z = input_points.cpu().numpy().T
            ax_input.scatter(x, y, z, c='green', s=1)
            ax_input.set_xlabel('X axis')
            ax_input.set_ylabel('Y axis')
            ax_input.set_zlabel('Z axis')
            ax_input.set_title('Input Points')

    plt.show()

def visualizeFeatureMap3dO3d(feature_map, output_dir, batch_idx=0, fmap_idx=None, input_points=None, same_plot=False, gt_boxes=None, pred_boxes=None):
    """Visualizes 3D feature maps using matplotlib.

    Args:
    feature_map [batch_size, feature_maps, z, y, x]: The feature map tensor to visualize.
    output_dir (str): The directory to save the visualization images.
    batch_idx (int): The index of the batch to visualize. Default is 0.
    fmap_idx (int): The index of the feature map to visualize. Default is None, which means all are being visualized.

    Raises:
    ValueError: If the feature_map is None or the output_dir does not exist.
    """
    if feature_map is None:
        raise ValueError("No feature map available. Check if the hook was triggered correctly.")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory `{output_dir}` does not exist.")

    # File operations
    feature_map = feature_map.dense().detach()
    num_feature_maps = feature_map.shape[1]

    # Check if fmap_idx is provided as a list, if not and it is not None, make it a list
    if isinstance(fmap_idx, list):
        fmap_indices = fmap_idx
    elif fmap_idx is not None:
        fmap_indices = [fmap_idx]
    else:
        fmap_indices = range(num_feature_maps)
    # Check if passed indices are within bounds
    for idx in fmap_indices:
        if not (0 <= idx < num_feature_maps):
            raise ValueError(f"fmap_idx {idx} is out of bounds. It must be between 0 and {num_feature_maps - 1}")

    if batch_idx < 0 or batch_idx >= feature_map.shape[0]:
        raise ValueError(f"batch_idx is out of bounds. It must be between 0 and {feature_map.shape[0] - 1}")

    if input_points is not None:
        input_points = input_points[:, 1:4]  # Only use the xyz coordinates

    for fmap_idx in fmap_indices:
        # one plot for each feature map

        single_feature_map = feature_map[batch_idx, fmap_idx] # values[z,y,x]
        z, y, x = torch.nonzero(single_feature_map, as_tuple=True)
        # scale values for colors
        nonzero_values = single_feature_map[z, y, x].cpu().numpy()
        # create a tensor of size 3 x N with nonzero_values in the first row, the rest is 0
        colors = np.zeros((3, len(nonzero_values)))
        colors[0, :] = abs(nonzero_values)
        colors = colors / colors.max() * 2 # values between 0 and 1
        colors = np.clip(colors, 0, 1)  # Ensure values are within range [0, 1]
        colors = colors.astype(np.float64)  # Convert to float64, as Open3D expects

        x_meters = x.cpu().numpy() * 0.1 - 51.2  # voxel size and pc range
        y_meters = y.cpu().numpy() * 0.1 - 51.2
        z_meters = z.cpu().numpy() * 0.2 - 4.9

        points = o3d.utility.Vector3dVector(np.vstack((x_meters, y_meters, z_meters)).T)
        colors = o3d.utility.Vector3dVector(colors.T)

        feature_pcd = o3d.geometry.PointCloud()
        feature_pcd.points = points
        feature_pcd.colors = colors

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(feature_pcd, voxel_size=1)

        # Create Open3d Visualizer:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().point_size = 2.5
        vis.get_render_option().background_color = np.ones(3)

        # Input Points Visualization
        if input_points is not None and same_plot:
            input_point_cloud = o3d.geometry.PointCloud()
            input_point_cloud.points = o3d.utility.Vector3dVector(input_points.detach().cpu().numpy().astype(np.float64))
            input_point_cloud.paint_uniform_color([0, 1, 0])  # green
            # o3d.visualization.draw_geometries([voxel_grid, input_point_cloud],
            # window_name=f'3D Feature Map and Input Points - Batch: {batch_idx}, Feature Map: {fmap_idx}')
            vis.add_geometry(input_point_cloud)
            vis.add_geometry(voxel_grid)

            #test
            if pred_boxes is not None:
                vis, box3d_list = draw_box(vis, pred_boxes.cpu(), (1, 0, 0))
                print('Number of Pred-Boxes: ', pred_boxes.shape[0])
            if gt_boxes is not None:
                gt_angles = gt_boxes[:, 6:8].reshape((-1, 2))
                for i in range(gt_boxes.shape[0]):
                    gt_box = gt_boxes[i, :].reshape((1, 9))
                    vis, box3d_list = draw_box(vis, gt_box.cpu(), (0, 0, 1))
                print('Number of GT-Boxes: ', gt_boxes.shape[0])
        elif input_points is not None and not same_plot:
            # o3d.visualization.draw_geometries([voxel_grid],
            #                                   window_name=f'3D Feature Map - Batch: {batch_idx}, Feature Map: {fmap_idx}')
            vis.add_geometry(voxel_grid)

            input_point_cloud = o3d.geometry.PointCloud()
            input_point_cloud.points = o3d.utility.Vector3dVector(input_points.detach().cpu().numpy().astype(np.float64))
            input_point_cloud.paint_uniform_color([0, 1, 0])  # green
            # o3d.visualization.draw_geometries([input_point_cloud],
            #                                   window_name='Input Points')
            vis.create_window()
            vis.add_geometry(input_point_cloud)
        else:  # no input points were passed
            vis.add_geometry(voxel_grid)
        vis.run()
        vis.destroy_window()


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