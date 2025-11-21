import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Path to the npz files
npz_dir = "../bloom/diff_frames"
npz_files = sorted(glob.glob(os.path.join(npz_dir, "diff_*.npz")), 
                   key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

print(f"Found {len(npz_files)} npz files")

# Global variables for navigation
current_index = 0
neighborhood_size = 20  # Adjust this to change zoom level
cbars = []  # Store colorbar references

# Create figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event))

def load_frame(index):
    """Load and return the data from a specific frame index."""
    if 0 <= index < len(npz_files):
        file_path = npz_files[index]
        data = np.load(file_path)["arr"]
        return data, file_path
    return None, None

def update_display():
    """Update the display with the current frame."""
    global cbars
    data, file_path = load_frame(current_index)
    if data is None:
        return
    
    # Remove old colorbars
    for cbar in cbars:
        cbar.remove()
    cbars = []
    
    # Clear axes
    ax1.clear()
    ax2.clear()
    
    # Find the maximum value and its location
    max_val = np.max(data)
    max_idx = np.unravel_index(np.argmax(data), data.shape)
    
    # Print info to console
    frame_num = os.path.basename(file_path).split('_')[1].split('.')[0]
    print(f"\nFrame {frame_num}: Shape={data.shape}, Max={max_val:.4f} at {max_idx}")
    
    # Full plot
    im1 = ax1.imshow(data, cmap='viridis', aspect='auto')
    cbar1 = plt.colorbar(im1, ax=ax1, label='Value')
    cbars.append(cbar1)
    ax1.set_title(f"PSF Array (2D) - Full View - Frame {frame_num}/{len(npz_files)-1}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    
    # Zoomed plot
    y_start = max(0, max_idx[0] - neighborhood_size)
    y_end = min(data.shape[0], max_idx[0] + neighborhood_size + 1)
    x_start = max(0, max_idx[1] - neighborhood_size)
    x_end = min(data.shape[1], max_idx[1] + neighborhood_size + 1)
    zoomed_region = data[y_start:y_end, x_start:x_end]
    
    im2 = ax2.imshow(zoomed_region, cmap='viridis', aspect='auto', 
                     extent=[x_start, x_end-1, y_end-1, y_start])
    cbar2 = plt.colorbar(im2, ax=ax2, label='Value')
    cbars.append(cbar2)
    ax2.plot(max_idx[1], max_idx[0], 'r*', markersize=15, 
             label=f'Max at ({max_idx[0]}, {max_idx[1]})')
    ax2.set_title(f"Zoomed View (Max at ({max_idx[0]}, {max_idx[1]}))")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.legend()
    
    plt.tight_layout()
    fig.canvas.draw()

def on_key(event):
    """Handle key press events."""
    global current_index
    
    if event.key == 'right' or event.key == 'd':
        # Next frame
        if current_index < len(npz_files) - 1:
            current_index += 1
            update_display()
    elif event.key == 'left' or event.key == 'a':
        # Previous frame
        if current_index > 0:
            current_index -= 1
            update_display()
    elif event.key == 'escape':
        # Close window
        plt.close()

# Initial display
update_display()

# Instructions
print("\n" + "="*60)
print("Navigation Controls:")
print("  Left Arrow / 'a'  : Previous frame")
print("  Right Arrow / 'd' : Next frame")
print("  Escape            : Close window")
print("="*60)

plt.show()
