import torch
import numpy as np
import matplotlib.pyplot as plt

def extract_feature_map_hook(module, input, output):
    """Extracts the output of the layer for visualization."""
    
    feature_maps = output

def register_hook_for_layer(model, layer_name):
    """Registers a hook for the specified layer in the model."""
    for name, module in model.named_modules():
        if name == layer_name:
            # Register a forward hook on the layer to capture feature maps
            module.register_forward_hook(extract_feature_map_hook)

def visualize_feature_map(feature_map, map_index, slice_index):
    """Visualizes a slice of one feature map using Matplotlib."""
    if feature_map is not None:
        # Detach the feature map from GPU and convert to NumPy
        feature_slice = feature_map[map_index][slice_index].cpu().detach().numpy()
        plt.imshow(feature_slice, cmap='viridis')
        plt.colorbar()
        plt.show()
    else:
        print("No feature map available. Check if the hook was triggered correctly.")