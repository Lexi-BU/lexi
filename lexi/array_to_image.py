import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path


def array_to_image(
    input_array: np.ndarray,
    x_range: list = None,
    y_range: list = None,
    v_min: float = None,
    v_max: float = None,
    cmap: str = "viridis",
    norm: mpl.colors.Normalize = None,
    norm_type: str = "linear",
    aspect: str = "auto",
    figure_title: str = None,
    show_colorbar: bool = True,
    cbar_label: str = None,
    cbar_orientation: str = "vertical",
    show_axes: bool = True,
    display: bool = False,
    figure_size: tuple = (10, 10),
    figure_format: str = "png",
    figure_font_size: float = 12,
    save: bool = False,
    save_path: str = None,
    save_name: str = None,
    dpi: int = 300,
    dark_mode: bool = False,
):
    """
    Convert a 2D array to an image.

    Parameters
    ----------
    input_array : np.ndarray
        2D array to convert to an image.
    x_range : list, optional
        Range of the x-axis.  Default is None.
    y_range : list, optional
        Range of the y-axis.  Default is None.
    v_min : float, optional
        Minimum value of the colorbar.  If None, then the minimum value of the input array is used.
        Default is None.
    v_max : float, optional
        Maximum value of the colorbar.  If None, then the maximum value of the input array is used.
        Default is None.
    cmap : str, optional
        Colormap to use.  Default is 'viridis'.
    norm : mpl.colors.Normalize, optional
        Normalization to use for the colorbar colors.  Default is None.
    norm_type : str, optional
        Normalization type to use.  Options are 'linear' or 'log'.  Default is 'linear'.
    aspect : str, optional
        Aspect ratio to use.  Default is 'auto'.
    figure_title : str, optional
        Title of the figure.  Default is None.
    show_colorbar : bool, optional
        If True, then show the colorbar.  Default is True.
    cbar_label : str, optional
        Label of the colorbar.  Default is None.
    cbar_orientation : str, optional
        Orientation of the colorbar.  Options are 'vertical' or 'horizontal'.  Default is 'vertical'.
    show_axes : bool, optional
        If True, then show the axes.  Default is True.
    display : bool, optional
        If True, then display the figure.  Default is False.
    figure_size : tuple, optional
        Size of the figure.  Default is (10, 10).
    figure_format : str, optional
        Format of the figure.  Default is 'png'.
    figure_font_size : float, optional
        Font size of the figure.  Default is 12.
    save : bool, optional
        If True, then save the figure.  Default is False.
    save_path : str, optional
        Path to save the figure to.  Default is None.
    save_name : str, optional
        Name of the figure to save.  Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes object.
    """
    # Try to use latex rendering
    try:
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        plt.rc("font", size=figure_font_size)
    except Exception:
        pass

    # Check whether input_array is a 2D array
    if len(input_array.shape) != 2:
        raise ValueError("input_array must be a 2D array")

    # Check whether x_range is a list
    if x_range is not None:
        if not isinstance(x_range, list):
            raise ValueError("x_range must be a list")
        if len(x_range) != 2:
            raise ValueError("x_range must be a list of length 2")
    else:
        x_range = [0, 360]

    # Check whether y_range is a list
    if y_range is not None:
        if not isinstance(y_range, list):
            raise ValueError("y_range must be a list")
        if len(y_range) != 2:
            raise ValueError("y_range must be a list of length 2")
    else:
        y_range = [-90, 90]

    # Check whether input_dict is a dictionary

    if dark_mode:
        plt.style.use("dark_background")
        facecolor = "k"
        edgecolor = "w"
    else:
        plt.style.use("default")
        facecolor = "w"
        edgecolor = "k"

    if v_min is None and v_max is None:
        if norm_type == "linear":
            v_min = 0.9 * np.nanmin(input_array)
            v_max = 1.1 * np.nanmax(input_array)
            norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
        elif norm_type == "log":
            v_min = np.nanmin(input_array)
            if v_min <= 0:
                v_min = 1e-10
            v_max = np.nanmax(input_array)
            norm = mpl.colors.LogNorm(vmin=v_min, vmax=v_max)
    elif v_min is not None and v_max is not None:
        if norm_type == "linear":
            norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
        elif norm_type == "log":
            if v_min <= 0:
                v_min = 1e-10
            norm = mpl.colors.LogNorm(vmin=v_min, vmax=v_max)
    else:
        raise ValueError(
            "Either both v_min and v_max must be specified or neither can be specified"
        )

    # Create the figure
    fig, ax = plt.subplots(
        figsize=figure_size, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor
    )

    # Plot the image
    im = ax.imshow(
        np.transpose(input_array),
        cmap=cmap,
        norm=norm,
        extent=[
            x_range[0],
            x_range[1],
            y_range[0],
            y_range[1],
        ],
        origin="lower",
        aspect=aspect,
    )

    # Set the tick label size
    ax.tick_params(labelsize=0.8 * figure_font_size)

    if show_colorbar:
        if cbar_label is None:
            cbar_label = "Value"
        if cbar_orientation == "vertical":
            cax = fig.add_axes(
                [
                    ax.get_position().x1 + 0.01,
                    ax.get_position().y0,
                    0.02,
                    ax.get_position().height,
                ]
            )
        elif cbar_orientation == "horizontal":
            cax = fig.add_axes(
                [
                    ax.get_position().x0,
                    ax.get_position().y1 + 0.01,
                    ax.get_position().width,
                    0.02,
                ]
            )
        ax.figure.colorbar(
            im,
            cax=cax,
            orientation=cbar_orientation,
            label=cbar_label,
            pad=0.01,
        )
        # Set the colorbar tick label size
        cax.tick_params(labelsize=0.6 * figure_font_size)
        # Set the colorbar label size
        cax.yaxis.label.set_size(0.9 * figure_font_size)

        # If the colorbar is horizontal, then set the location of the colorbar label and the tick
        # labels to be above the colorbar
        if cbar_orientation == "horizontal":
            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")
            cax.xaxis.tick_top()
        if cbar_orientation == "vertical":
            cax.yaxis.set_ticks_position("right")
            cax.yaxis.set_label_position("right")
            cax.yaxis.tick_right()
    if not show_axes:
        ax.axis("off")
    else:
        ax.set_xlabel("RA [$^\\circ$]", labelpad=0, fontsize=figure_font_size)
        ax.set_ylabel("DEC [$^\\circ$]", labelpad=0, fontsize=figure_font_size)
        ax.set_title(figure_title, fontsize=1.2 * figure_font_size)

    if save:
        if save_path is None:
            save_path = Path(__file__).resolve().parent.parent / "figures"
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        if save_name is None:
            save_name = "array_to_image"

        save_name = save_name + "." + figure_format
        plt.savefig(
            save_path / save_name, format=figure_format, dpi=dpi, bbox_inches="tight"
        )
        print(f"Saved figure to {save_path / save_name}")

    if display:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    # Create a test array
    test_array = np.random.rand(100, 100)

    # Plot the test array
    _ = array_to_image(
        input_array=test_array,
        x_range=[0, 360],
        y_range=[-90, 90],
        display=False,
        save=True,
        save_name="test_array_to_image",
        figure_format="pdf",
        dark_mode=False,
        figure_font_size=16,
        cbar_orientation="vertical",
        figure_title="Test Array",
    )
