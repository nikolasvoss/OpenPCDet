import os
import torch
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

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

    feature_map = feature_map.dense().detach()

    # Prepare to visualize only specified feature maps or z-planes
    num_feature_maps = feature_map.size(1)
    num_z_planes = feature_map.size(2)

    feature_map_indices = [fmap_idx] if fmap_idx is not None else range(num_feature_maps)
    z_plane_indices = [z_plane] if z_plane is not None else range(num_z_planes)

    # Check index bounds
    if batch_idx < 0 or batch_idx >= feature_map.size(0):
        raise ValueError(f"batch_idx is out of bounds. It must be between 0 and {feature_map.size(0) - 1}")
    if fmap_idx and (fmap_idx < 0 or fmap_idx >= num_feature_maps):
        raise ValueError(f"fmap_idx is out of bounds. It must be between 0 and {num_feature_maps - 1}")
    if z_plane and (z_plane < 0 or z_plane >= num_z_planes):
        raise ValueError(f"z_plane is out of bounds. It must be between 0 and {num_z_planes - 1}")


    visibility_factor = 4  # multiply all values for better visibility

    # Visualization loop
    for fmap_idx in feature_map_indices:
        for z_plane in z_plane_indices:
            # Extract the specific slice to visualize
            feature_slice = feature_map[batch_idx, fmap_idx, z_plane].cpu().numpy()

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

        # Save the plot as an image
        # file_name = f'map3D_batch{batch_idx}_fmap{fmap_idx}.png'
        # plt.savefig(os.path.join(output_dir, file_name), dpi=150)
        # plt.close(fig)