from typing import NoReturn, Optional, Tuple, Callable, Union, Generator

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

from ._utils import MplColorHelper


def plot_pet_shap(
    image_path: str,
    masks_dict: dict,
    save_path: str,
    alpha: Optional[float] = 0.7,
    max_suv: Optional[float] = 10,
    cmap_name: Optional[str] = "seismic",
    cmap_lim: Optional[Tuple[float, float]] = (0, 0.5),
    centered_norm: Optional[bool] = True,
    plot_colorbar: Optional[bool] = True,
    title: Optional[Union[None, str]] = None,
) -> None:
    """
    Parameters
    ----------
    image_path:

    masks_dict:

    save_path:

    alpha:

    max_suv:

    cmap_name:

    cmap_lim:

    centered_norm:

    plot_colorbar:

    title:

    Returns
    -------
    None
    """
    # Load image and create MIP views
    img = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(img).astype(np.float32)  # [-450:, :, :]
    mip_1 = np.flipud(np.max(np.rot90(img_array, k=1, axes=(2, 1)), axis=-1))
    mip_2 = np.flipud(np.max(img_array, axis=-1))

    # Define color map
    cmap = MplColorHelper(
        cmap_name=cmap_name, cmap_lim=cmap_lim, centered_norm=centered_norm
    )

    # Plot MIP views
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    axes[0].imshow(mip_1, cmap="binary", vmin=0, vmax=max_suv)
    axes[1].imshow(mip_2, cmap="binary", vmin=0, vmax=max_suv)

    for mask_path, importance in masks_dict.items():
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
        cbar_ax = fig.add_axes([0.91, 0.2, 0.03, 0.6])
        plt.colorbar(mappable=cmap.scalarMap, cax=cbar_ax)
    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.15)
    fig.savefig(save_path)
    return
