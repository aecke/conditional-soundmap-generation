import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd



def compute_visibility(osm_arr):
    """Compute LoS and NLoS masks."""
    from numba import jit
    
    @jit(nopython=True)
    def ray_tracing(image_size, image_map):
        visibility_map = np.zeros((image_size, image_size))
        source = (image_size // 2, image_size // 2)
        for x in range(image_size):
            for y in range(image_size):
                dx = x - source[0]
                dy = y - source[1]
                dist = np.sqrt(dx*dx + dy*dy)
                steps = int(dist)
                if steps == 0:
                    continue
                step_dx = dx / steps
                step_dy = dy / steps

                visible = True
                ray_x, ray_y = source
                for _ in range(steps):
                    ray_x += step_dx
                    ray_y += step_dy
                    int_x, int_y = int(ray_x), int(ray_y)
                    if 0 <= int_x < image_size and 0 <= int_y < image_size:
                        if image_map[int_y, int_x] == 0:
                            visible = False
                            break
                visibility_map[y, x] = visible
        return visibility_map

    # Ensure binary image map
    image_map = np.where(osm_arr > 0, 1, 0)
    visibility_map = ray_tracing(osm_arr.shape[0], image_map)
    
    # Calculate visibility masks
    pixels_in_sight = np.logical_and(visibility_map == 1, image_map == 1)
    pixels_not_in_sight = np.logical_and(visibility_map == 0, image_map == 1)
    
    # Apply building mask
    pixels_not_in_sight = np.where(image_map == 0, 0, pixels_not_in_sight)
    pixels_in_sight = np.where(image_map == 0, 0, pixels_in_sight)
    
    return pixels_in_sight, pixels_not_in_sight

def plot_soundmaps(index, pred_path, test_dir):
    """
    Plot soundmaps and error maps (LoS/NLoS) for a given index with publication-quality formatting.
    """
    # Set font sizes
    TITLE_SIZE = 28
    LABEL_SIZE = 20
    COLORBAR_LABEL_SIZE = 20
    COLORBAR_TICK_SIZE = 18
    
    # Set default font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': LABEL_SIZE
    })
    
    # Load test data info
    test_csv = pd.read_csv(os.path.join(test_dir, "test.csv"))
    sample = test_csv.iloc[index]
    
    # Load prediction (ensure exact 256x256)
    pred_img = Image.open(os.path.join(pred_path, f"y_{index}.png")).convert("L")
    pred_img = pred_img.resize((256, 256), Image.Resampling.NEAREST)
    pred_soundmap = 100 - (np.array(pred_img, dtype=np.float32)) / 255 * 100

    # Load true soundmap
    true_img = Image.open(os.path.join(test_dir, "soundmaps/256", os.path.basename(sample.soundmap))).convert("L")
    true_img = true_img.resize((256, 256), Image.Resampling.NEAREST)
    true_soundmap = 100 - (np.array(true_img, dtype=np.float32)) / 255 * 100

    # Load and resize building map
    osm = np.array(
        Image.open(os.path.join(test_dir, "buildings", os.path.basename(sample.osm)))
        .convert("L")
        .resize((256, 256), Image.Resampling.NEAREST),
        dtype=np.int16
    )
    
    # Calculate visibility masks
    pixels_in_sight, pixels_not_in_sight = compute_visibility(osm)
    
    # Calculate error maps
    error_map = np.abs(true_soundmap - pred_soundmap)
    
    # Create masked error maps
    los_error = np.where(pixels_in_sight, error_map, np.nan)
    nlos_error = np.where(pixels_not_in_sight, error_map, np.nan)

    # Create figure and axes with specific size
    fig = plt.figure(figsize=(24, 8))
    
    # Create main subplot area
    gs = plt.GridSpec(1, 4, figure=fig)
    gs.update(left=0.02, right=0.85, top=0.85, bottom=0.15, wspace=0.05)
    
    axes = []
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)

    # Plot soundmaps
    im1 = axes[0].imshow(true_soundmap, cmap='viridis')
    im2 = axes[1].imshow(pred_soundmap, cmap='viridis')
    
    # Plot error maps with individual scaling
    vmax_los = np.nanmax(los_error)
    vmax_nlos = np.nanmax(nlos_error)
    
    im3 = axes[2].imshow(los_error, cmap='viridis', vmin=0, vmax=vmax_los)
    im4 = axes[3].imshow(nlos_error, cmap='viridis', vmin=0, vmax=vmax_nlos)

    # Add titles with max error values
    titles = [
        'True Soundmap',
        'Predicted Soundmap',
        f'LoS Error (max: {vmax_los:.1f}dB)',
        f'NLoS Error (max: {vmax_nlos:.1f}dB)'
    ]

    # Add building overlay and clean up axes
    for ax, title in zip(axes, titles):
        # Add building overlay
        red_overlay = np.zeros((*osm.shape, 4), dtype=np.uint8)
        red_overlay[osm == 0, 0] = 255  # Red channel
        red_overlay[osm == 0, 3] = 255  # Alpha channel
        ax.imshow(red_overlay)
        
        # Clean up axes
        ax.set_title(title, pad=20, fontsize=TITLE_SIZE)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Add soundmap colorbar
    cbar_ax = fig.add_axes([0.87, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='vertical')
    cbar.set_label('decibels (dB)', fontsize=COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    # Add error colorbars
    cbar_ax2 = fig.add_axes([0.91, 0.15, 0.015, 0.32])  # Lower half
    cbar2 = fig.colorbar(im3, cax=cbar_ax2, orientation='vertical')
    cbar2.set_label('LoS error (dB)', fontsize=COLORBAR_LABEL_SIZE)
    cbar2.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    cbar_ax3 = fig.add_axes([0.91, 0.53, 0.015, 0.32])  # Upper half
    cbar3 = fig.colorbar(im4, cax=cbar_ax3, orientation='vertical')
    cbar3.set_label('NLoS error (dB)', fontsize=COLORBAR_LABEL_SIZE)
    cbar3.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    return fig

if __name__ == "__main__":
    # Configuration
    PRED_DIR = "E:/ba_ergebnisse/urban_sound_25k_combined/evaluation_results/urban_sound_25k_combined/predictions"
    TEST_DIR = "E:/Schallsimulationsdaten/urban_sound_25k_combined/test"
    OUTPUT_DIR = "E:/ba_ergebnisse/urban_sound_25k_combined/evaluation_results/urban_sound_25k_combined/sample_visualizations"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Plot first sample (index 0)
    sample_idx = 0
    fig = plot_soundmaps(sample_idx, PRED_DIR, TEST_DIR)
    
    # Save plot with minimal borders and high DPI
    output_path = os.path.join(OUTPUT_DIR, f"sample_{sample_idx}_comparison.png")
    fig.savefig(output_path, 
                bbox_inches='tight', 
                pad_inches=0.2,
                dpi=300)
    plt.close(fig)
    
    print(f"Visualization saved to: {output_path}")