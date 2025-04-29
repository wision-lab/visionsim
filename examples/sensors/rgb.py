from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from natsort import natsorted

from visionsim.emulate.rgb import emulate_rgb_from_sequence

# Read all images, normalized to [0, 1] range
imgs = [iio.imread(p).astype(float) / 255 for p in natsorted(Path("lego-interp/").glob("*.png"))]

# Emulate RGB camera by varying exposure, with no noise
blur = [emulate_rgb_from_sequence(imgs[:n], readout_std=0, fwc=n) for n in (10, 40, 160, 640)]

# Emulate RGB camera by varying noise, with fixed blur
noise = [emulate_rgb_from_sequence(imgs[:80], readout_std=n, fwc=80) for n in (80, 40, 20, 10)]

# Plot result
fig = plt.figure()
grid = ImageGrid(fig, (1, 1, 1), nrows_ncols=(2, 4))

for ax, img in zip(grid, blur + noise):
    ax.set_axis_off()
    ax.imshow(img)
plt.show()
