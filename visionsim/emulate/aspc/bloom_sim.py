# 2D Bloom simulation (copy-paste and run)
import numpy as np
import matplotlib.pyplot as plt

def simulate_bloom_2d(H=128, W=128, center=None,
                      saturated_value=1.0,
                      alpha=0.35,
                      decay_rate=0.3,
                      elongation_x_left=1.0, elongation_x_right=1.0,
                      elongation_y_up=1.0, elongation_y_down=1.0,
                      max_extent=6,
                      intensity_scale=1.0,
                      cmap="viridis"):
    """
    Simulate 2D bloom from a single saturated "aggressor" pixel.

    Parameters
    ----------
    H,W : int
        image size
    center : tuple or None
        (cx,cy) center; if None uses image center
    saturated_value : float
        value at aggressor before bloom
    alpha : float in [0,1]
        fraction kept by aggressor (the rest redistributes)
    decay_rate : float
        exponential decay rate for circular/ellipsoidal PSF (larger -> faster decay)
    elongation_x_left, elongation_x_right : float
        elongation factors for x direction (left/negative and right/positive)
        (1.0 = circular, >1.0 = elongated)
    elongation_y_up, elongation_y_down : float
        elongation factors for y direction (up/negative and down/positive)
        (1.0 = circular, >1.0 = elongated)
    max_extent : int
        truncate PSF beyond this pixel radius (default: 6)
    intensity_scale : float
        scale for redistributed energy (relative to saturated_value)
    cmap : str
        matplotlib colormap for display

    Returns
    -------
    clean, observed, psf : numpy arrays (H,W)
    """
    if center is None:
        cx, cy = W // 2, H // 2
    else:
        cy, cx = center

    # Clean image
    clean = np.zeros((H, W), dtype=float)
    clean[cy, cx] = saturated_value

    # coordinate grids
    y = np.arange(H)
    x = np.arange(W)
    xx, yy = np.meshgrid(x, y)
    dx = xx - cx
    dy = yy - cy


    # Select elongation factors based on direction
    # For x: left (dx < 0) uses elongation_x_left, right (dx >= 0) uses elongation_x_right
    elongation_x_effective = np.where(dx < 0, elongation_x_left, elongation_x_right)
    # For y: up (dy < 0) uses elongation_y_up, down (dy >= 0) uses elongation_y_down
    elongation_y_effective = np.where(dy < 0, elongation_y_up, elongation_y_down)

    # Calculate ellipsoidal distance with direction-dependent elongation
    # distance = sqrt((dx/elongation_x_effective)^2 + (dy/elongation_y_effective)^2)
    ellipsoid_dist = np.sqrt((dx / elongation_x_effective)**2 + (dy / elongation_y_effective)**2)

    # Circular/ellipsoidal PSF with exponential decay
    psf = np.exp(-ellipsoid_dist * decay_rate)

    # truncate beyond max_extent pixels (circular/ellipsoidal mask)
    mask_extent = ellipsoid_dist <= max_extent
    psf *= mask_extent

    # remove center (center is the aggressor; its retained energy not part of redistributed kernel)
    psf[cy, cx] = 0.0

    # # normalize PSF (so its sum = 1)
    # s = psf.sum()
    # if s <= 0:
    #     raise ValueError("PSF sum is zero; increase max_extent or reduce decay rates.")
    # psf /= s

    # redistributed energy map = (1-alpha) * saturated_value * intensity_scale * psf
    redistributed = (1.0 - alpha) * saturated_value * intensity_scale * psf

    # observed: center keeps alpha * sat, plus redistributed everywhere
    observed = np.copy(clean)
    observed[cy, cx] = alpha * saturated_value
    observed += redistributed

    return clean, observed, psf

def plot_results(clean, observed, psf, cmap="viridis"):
    plt.figure(figsize=(14,4))
    ax = plt.subplot(1,3,1)
    plt.title("Clean (single saturated pixel)")
    im0 = plt.imshow(clean, origin='lower', cmap=cmap, vmin=0, vmax=1.0)
    plt.colorbar(im0, fraction=0.046, pad=0.04)
    cy, cx = np.array(clean.shape) // 2
    plt.scatter([cx], [cy], c='red', s=10)

    plt.subplot(1,3,2)
    plt.title("Observed (with 2D bloom)")
    # set vmax small to help visualize tails; adjust as needed
    im1 = plt.imshow(observed, origin='lower', cmap=cmap, vmax=max(0.6, observed.max()))
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.scatter([cx], [cy], c='red', s=10)

    plt.subplot(1,3,3)
    plt.title("Normalized PSF (used for redistribution)")
    # show PSF (log-like visibility); add small eps to avoid log(0)
    im2 = plt.imshow(psf, origin='lower', cmap='inferno')
    plt.colorbar(im2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage with asymmetric ellipsoidal PSF
    clean, observed, psf = simulate_bloom_2d(
        H=192, W=256,
        center=(96, 128),
        saturated_value=5.0,
        alpha=0.50,               # 50% stays in aggressor, 50% redistributes
        decay_rate=0.5,           # exponential decay rate
        elongation_x_left=1.5,   # left direction elongation
        elongation_x_right=1.0,   # right direction elongation
        elongation_y_up=1.0,      # up direction elongation
        elongation_y_down=1.5,    # down direction elongation
        max_extent=4,             # PSF decays within 6 pixels
        intensity_scale=1.0
    )

    plot_results(clean, observed, psf)
