from typing import NoReturn, Optional, Tuple, Callable, Union, Generator

import SimpleITK as sitk
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._utils import MplColorHelper


def plot_bars(
    shapley_values: np.ndarray,
    nbest: Optional[int] = 10,
    names: Optional[Union[list, None]] = None,
    sort: Optional[bool] = True,
    ax: Optional[Union[matplotlib.axes.Axes, None]] = None,
) -> None:
    """Plot the Shapley values of different instances with a bar plot.

    Parameters
    ----------
    shapley_values : 1D array shape (n_shapley_values,)
        Shapley values.

    nbest : int, optional.
        Number of values to display on the bar plot. The nbest features with the highest Shapley values (absolute value)
        will be displayed. If the tolal number of regions is lower than nbest all the values will be displayed. Only
        valid when `sort` is True. The default is 10.

    names : list of strings or None, optional.
        Names of the regions associated to the Shapley values. If None default names 'instance_0', ..., 'instance_n'
        will be used. The default is None.

    sort : bool, optional.
        If true Shapley values are ranked from top to bottom in decreasing order with respect to their absolute value.
        If False all the regions will be displayed on the bar plot. The default is True.

    ax : matplotlib.axes, optional
            The default is None.

    Returns
    -------
    None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    if names is not None:
        df = pd.DataFrame(shapley_values, index=names, columns=["shapley"])
    else:
        df = pd.DataFrame(
            shapley_values,
            index=["instance_" + str(i) for i in range(len(shapley_values))],
            columns=["shapley"],
        )

    df["shapley_abs"] = np.abs(df["shapley"])
    df["sign"] = 1 * (df["shapley"] > 0)
    if sort:
        df = df.sort_values(by="shapley_abs", ascending=False).iloc[
            : max(nbest, len(shapley_values)), :
        ]

    sns.barplot(
        data=df.reset_index(),
        orient="h",
        x="shapley",
        y="index",
        hue="sign",
        hue_order=[0, 1],
        palette=["blue", "red"],
        dodge=False,
        ax=ax,
    )
    ax.set(xlabel=None, ylabel=None)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="gray", linestyle="dashed")
    ax.legend().set_visible(False)
    ax.axvline(x=0, color="k", linestyle="--")
    # ax.set_xticks(
    #     ticks=ax.get_xticks(), labels=np.round(ax.get_xticks() + shap.empty_value, 4)
    # )
    sns.despine()
    return


def plot_pet(
    shapley_values: np.ndarray,
    image_path: str,
    masks_paths: list,
    save_path: Optional[Union[None, str]] = None,
    alpha: Optional[float] = 0.7,
    max_suv: Optional[float] = 10,
    cmap_name: Optional[str] = "seismic",
    cmap_lim: Optional[Union[None, float]] = None,
    plot_colorbar: Optional[bool] = True,
    title: Optional[Union[None, str]] = None,
) -> None:
    """Display the Shapley values of different instances on the Maxumum Intensity Projection (MIP) of a PET image.
    The different values are represented with a color scale.

    Parameters
    ----------
    shapley_values : 1D array shape (n_shapley_values,)
        Shapley values.

    image_path: string.
        Path of the PET image. The PET image format should be compatible with simpleitk. Nifti images (.nii.gz) are
        recommended

    masks_paths: list of strings.
        List of paths for the masks of the regions of interest associated with the Shapley values. Nifti images
        (.nii.gz) are recommended

    save_path: string or None, optional.
        Path to save the plot. If None the plotis not saved.
        The default is None.

    alpha: float in [0, 1], optional.
        Parameter to regulate the transparancy of the colored masks.
        The default is 0.7

    max_suv: float, optional
        Upper limit for the binary color mapping of the SUV intensities.
        The default is 10.

    cmap_name: string, optional
        The default is 'seismic'.

    cmap_lim: float or None, optional
        The default is None.

    plot_colorbar: bool, optional
        The default is True.

    title: string or None, optional
        Title of the plot. If None no title will be displayed.
        The default is None.

    Returns
    -------
    None.
    """
    # Load image and create MIP views
    img = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(img).astype(np.float32)  # [-450:, :, :]
    mip_1 = np.flipud(np.max(np.rot90(img_array, k=1, axes=(2, 1)), axis=-1))
    mip_2 = np.flipud(np.max(img_array, axis=-1))

    # shap_values = shap.shapleyvalues_.copy() + shap.empty_value
    # Define color map
    if cmap_lim is None:
        cmap = MplColorHelper(
            cmap_name=cmap_name,
            cmap_lim=(0, np.abs(shapley_values).max()),
            # cmap_lim=(shap.empty_value, np.abs(shap.shapleyvalues_).max()),
        )
    else:
        cmap = MplColorHelper(
            cmap_name=cmap_name,
            cmap_lim=(0, cmap_lim)
            # cmap_name=cmap_name, cmap_lim=(shap.empty_value, cmap_lim)
        )

    # Plot MIP views
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    axes[0].imshow(mip_1, cmap="binary", vmin=0, vmax=max_suv)
    axes[1].imshow(mip_2, cmap="binary", vmin=0, vmax=max_suv)

    for mask_path, importance in zip(masks_paths, shapley_values):
        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask).astype(np.float32)  # [-450:, :, :]
        mask_mip_1 = np.flipud(np.max(np.rot90(mask_array, k=1, axes=(2, 1)), axis=-1))
        mask_mip_2 = np.flipud(np.max(mask_array, axis=-1))
        color = cmap.get_rgb(importance, alpha=alpha)

        mask_1 = np.zeros((mask_mip_1.shape[0], mask_mip_1.shape[1], 4))
        mask_2 = np.zeros((mask_mip_2.shape[0], mask_mip_2.shape[1], 4))
        mask_1[mask_mip_1 == 1, :] = color
        mask_2[mask_mip_2 == 1, :] = color

        axes[0].imshow(mask_1)
        axes[1].imshow(mask_2)

    axes[0].axis("off")
    axes[1].axis("off")
    if plot_colorbar:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
        cb = fig.colorbar(mappable=cmap.scalarMap, aspect=80, cax=cbar_ax)
        cbar_ax.yaxis.set_label_position("left")
        cb.set_label("Shapley values", size=12, labelpad=5)
    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.15)
    if save_path is not None:
        fig.savefig(save_path)
    return
