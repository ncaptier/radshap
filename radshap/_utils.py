import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl


class MplColorHelper:
    def __init__(self, cmap_name, cmap_lim, centered_norm=True):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        if centered_norm:
            self.norm = mpl.colors.CenteredNorm(
                vcenter=cmap_lim[0], halfrange=cmap_lim[1], clip=False
            )
        else:
            self.norm = mpl.colors.Normalize(vmin=cmap_lim[0], vmax=cmap_lim[1])
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val, alpha):
        return self.scalarMap.to_rgba(val, alpha=alpha)
