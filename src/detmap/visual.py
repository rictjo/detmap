import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgba

bUseHover=False
try:
    import mplcursors
    bUseHover = True
except ImportError :
    bUseHover = False
    print("warning hover functions need install of mplcursors library")
    mplcursors = None

def get_label_colors(labels, colormap='tab20', return_type='array',
                     categorical=True, seed=None):
    """
    Generate colors for labels using matplotlib colormaps.

    Parameters:
    -----------
    labels : array-like
        List or array of labels
    colormap : str or matplotlib colormap
        Name of colormap ('tab10', 'viridis', 'Set1', 'rainbow', etc.) or cmap object
    return_type : str
        'array' - returns array of colors matching input labels
        'dict' - returns dictionary mapping label -> color
        'both' - returns both
    categorical : bool
        If True, uses discrete colors (good for categorical labels)
        If False, uses continuous mapping (good for ordered labels)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    colors : array or dict
        Colors for each label or mapping
    """

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Get unique labels
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    # Get colormap
    if isinstance(colormap, str):
        cmap = plt.cm.get_cmap(colormap)
    else:
        cmap = colormap

    # Generate colors based on colormap type
    if categorical:
        if colormap in ['tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'Paired',
                        'Accent', 'Dark2', 'Pastel1', 'Pastel2']:
            # Discrete colormap with distinct colors
            colors_list = [cmap(i % cmap.N) for i in range(n_labels)]
        else:
            # For continuous colormaps, sample evenly spaced points
            if n_labels <= cmap.N:
                indices = np.linspace(0, cmap.N - 1, n_labels, dtype=int)
            else:
                indices = np.linspace(0, cmap.N - 1, n_labels, dtype=int)
            colors_list = [cmap(i) for i in indices]
    else:
        # Continuous mapping (normalize labels to 0-1 range)
        if all(isinstance(label, (int, float)) for label in unique_labels):
            # If labels are numeric
            label_array = np.array(unique_labels)
            norm = plt.Normalize(label_array.min(), label_array.max())
            colors_list = [cmap(norm(label)) for label in unique_labels]
        else:
            # If labels are categorical but want continuous look
            indices = np.linspace(0, 1, n_labels)
            colors_list = [cmap(i) for i in indices]

    # Create mapping dictionary
    label_to_color = dict(zip(unique_labels, colors_list))

    # Generate array of colors matching input labels
    color_array = [label_to_color[label] for label in labels]

    # Return based on return_type
    if return_type == 'array':
        return color_array
    elif return_type == 'dict':
        return label_to_color
    elif return_type == 'both':
        return color_array, label_to_color
    else:
        raise ValueError("return_type must be 'array', 'dict', or 'both'")


def plot_colored_points(x, y, labels, colormap='tab10', figsize=(10, 8),
                        alpha=0.7, s=50, add_legend=True, title=None):
    """
    Quick plotting function using the color generator.

    Parameters:
    -----------
    x, y : array-like
        Coordinates
    labels : array-like
        Labels for coloring
    colormap : str
        Matplotlib colormap name
    figsize : tuple
        Figure size
    alpha : float
        Transparency
    s : int or float
        Point size
    add_legend : bool
        Whether to add legend
    title : str
        Plot title
    """

    # Get colors and mapping
    colors, label_map = get_label_colors(labels, colormap, return_type='both')

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot points
    scatter = ax.scatter(x, y, c=colors, alpha=alpha, s=s)

    # Add legend if requested
    if add_legend:
        # Create custom legend handles
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=label_map[label],
                                 edgecolor='black',
                                 label=label)
                          for label in label_map.keys()]
        ax.legend(handles=legend_elements, title='Labels',
                  bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax

if True:
    def plot_colored_points_with_hover(x, y, labels, hover_texts=None, colormap='tab10',
                                   figsize=(10, 8), alpha=0.7, s=50, title=None):
        """
    Plot colored points with hover annotations.

    Parameters:
    -----------
    x, y : array-like
        Coordinates
    labels : array-like
        Labels for coloring
    hover_texts : array-like, optional
        Custom text for each point. If None, uses label information
    colormap : str
        Matplotlib colormap name
    figsize : tuple
        Figure size
    alpha : float
        Transparency
    s : int or float
        Point size
    title : str
        Plot title
        """

        # Get colors using your existing function
        colors, label_map = get_label_colors(labels, colormap, return_type='both')

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot points
        scatter = ax.scatter(x, y, c=colors, alpha=alpha, s=s)

        # Prepare hover texts
        if hover_texts is None:
            # Generate default hover texts
            hover_texts = []
            for i in range(len(x)):
                text = f'Label: {labels[i]}\nX: {x[i]:.3f}\nY: {y[i]:.3f}'
                if hasattr(x[i], '__round__'):
                    text = f'Label: {labels[i]}\nX: {round(x[i], 3)}\nY: {round(y[i], 3)}'
                hover_texts.append(text)

        # Add hover functionality
        cursor = mplcursors.cursor(scatter, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            index = sel.index
            sel.annotation.set_text(hover_texts[index])
            sel.annotation.get_bbox_patch().set(fc="lightyellow", alpha=0.9)
            sel.annotation.get_bbox_patch().set(boxstyle="round,pad=0.5")
            sel.annotation.set_ha('center')
            sel.annotation.set_fontsize(10)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=label_map[label],
                             edgecolor='black',
                             label=label)
                      for label in label_map.keys()]
        ax.legend(handles=legend_elements, title='Labels',
              bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if title:
            ax.set_title(title)

        plt.tight_layout()
        return fig, ax, cursor
#else:
#    def plot_colored_points_with_hover(x, y, labels, hover_texts=None, colormap='tab10',
#                                   figsize=(10, 8), alpha=0.7, s=50, title=None):
#        print("Not supported install mplcursors")
#        exit(1)



# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_points = 100
    x = np.random.randn(n_points)
    y = np.random.randn(n_points)
    labels = np.random.choice(['Group A', 'Group B', 'Group C', 'Group D'], n_points)

    # Example 1: Using tab10 (categorical colormap)
    colors = get_label_colors(labels, colormap='tab10', return_type='array')

    # Example 2: Using viridis with mapping dictionary
    label_colors = get_label_colors(labels, colormap='viridis', return_type='dict')
    print("Color mapping:", label_colors)

    # Example 3: Using Set1 with both outputs
    colors_arr, colors_dict = get_label_colors(labels, colormap='Set1', return_type='both')

    # Example 4: Using continuous colormap for ordered labels
    ordered_labels = ['Low', 'Medium', 'High'] * 33 + ['Low']  # 100 points
    colors_continuous = get_label_colors(ordered_labels, colormap='RdYlGn',
                                         categorical=False, return_type='array')

    # Quick plotting
    fig, ax = plot_colored_points(x, y, labels, colormap='tab10',
                                   title='Points Colored by Group')
    plt.show()

    # Example with different colormaps
    colormaps = ['tab10', 'Set2', 'viridis', 'rainbow', 'coolwarm']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, cmap_name in enumerate(colormaps):
        ax = axes[idx]
        colors = get_label_colors(labels, colormap=cmap_name, return_type='array')
        ax.scatter(x, y, c=colors, alpha=0.7, s=50)
        ax.set_title(f'Colormap: {cmap_name}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.tight_layout()
    plt.show()
