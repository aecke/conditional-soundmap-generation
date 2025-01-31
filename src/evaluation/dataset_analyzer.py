import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd
import argparse
from pathlib import Path

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

    image_map = np.where(osm_arr > 0, 1, 0)
    visibility_map = ray_tracing(osm_arr.shape[0], image_map)

    pixels_in_sight = np.logical_and(visibility_map == 1, image_map == 1)
    pixels_not_in_sight = np.logical_and(visibility_map == 0, image_map == 1)

    pixels_not_in_sight = np.where(image_map == 0, 0, pixels_not_in_sight)
    pixels_in_sight = np.where(image_map == 0, 0, pixels_in_sight)

    return pixels_in_sight, pixels_not_in_sight

def plot_soundmaps(sample_data, pred_path, test_dir, add_buildings=True):
    """
    Plot soundmaps and error maps (LoS/NLoS) for a given sample with publication-quality formatting.

    Args:
        sample_data: Row from the test CSV containing sample information
        pred_path: Path to the predictions directory
        test_dir: Path to the test data directory
    """
    # Set font sizes
    TITLE_SIZE = 28
    LABEL_SIZE = 20
    COLORBAR_LABEL_SIZE = 20
    COLORBAR_TICK_SIZE = 18

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': LABEL_SIZE
    })

    # Get sample index from the filename
    sample_idx = int(sample_data.name)

    # Load prediction
    pred_img = Image.open(os.path.join(pred_path, f"y_{sample_idx}.png")).convert("L")
    pred_img = pred_img.resize((256, 256), Image.Resampling.NEAREST)
    pred_soundmap = 100 - (np.array(pred_img, dtype=np.float32)) / 255 * 100

    # Load true soundmap
    true_img = Image.open(os.path.join(test_dir, "soundmaps/256", os.path.basename(sample_data.soundmap))).convert("L")
    true_img = true_img.resize((256, 256), Image.Resampling.NEAREST)
    true_soundmap = 100 - (np.array(true_img, dtype=np.float32)) / 255 * 100

    # Load building map
    osm = np.array(
        Image.open(os.path.join(test_dir, "buildings", os.path.basename(sample_data.osm)))
        .convert("L")
        .resize((256, 256), Image.Resampling.NEAREST),
        dtype=np.int16
    )

    # Calculate visibility masks
    pixels_in_sight, pixels_not_in_sight = compute_visibility(osm)

    # Calculate error maps
    error_map = np.abs(true_soundmap - pred_soundmap)

    # Create masked error maps without NaN clipping
    los_error = np.where(pixels_in_sight, error_map, 0)
    nlos_error = np.where(pixels_not_in_sight, error_map, 0)

    # Create figure with 5 subplots
    fig = plt.figure(figsize=(30, 8))

    # Update GridSpec for 5 subplots
    gs = plt.GridSpec(1, 5, figure=fig)
    gs.update(left=0.02, right=0.85, top=0.85, bottom=0.15, wspace=0.05)

    axes = []
    for i in range(5):  # Create 5 subplots
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)

    # Find global min/max for consistent scaling
    vmin_sound = min(np.min(true_soundmap), np.min(pred_soundmap))
    vmax_sound = max(np.max(true_soundmap), np.max(pred_soundmap))
    
    # Find the global min/max for the error maps, but only consider values greater than 0
    vmin_error = min(np.min(los_error[los_error > 0]), np.min(nlos_error[nlos_error > 0])) if np.any(los_error > 0) and np.any(nlos_error > 0) else 0
    vmax_error = max(np.max(los_error), np.max(nlos_error))

    # Create custom colormap for absolute error (green to yellow to red)
    error_colors = ['green', 'yellow', 'red']
    error_cmap = plt.cm.RdYlGn_r  # Red-Yellow-Green colormap, reversed

    # Plot soundmaps and error maps with unified scale
    im1 = axes[0].imshow(true_soundmap, cmap='viridis', vmin=vmin_sound, vmax=vmax_sound)
    im2 = axes[1].imshow(pred_soundmap, cmap='viridis', vmin=vmin_sound, vmax=vmax_sound)
    im3 = axes[2].imshow(los_error, cmap='viridis', vmin=vmin_error, vmax=vmax_error)
    im4 = axes[3].imshow(nlos_error, cmap='viridis', vmin=vmin_error, vmax=vmax_error)

    # Plot absolute error with custom colormap
    im5 = axes[4].imshow(error_map, cmap=error_cmap, vmin=0, vmax=np.max(error_map))

    titles = [
        'True Soundmap',
        'Predicted Soundmap',
        'LoS Error',
        'NLoS Error',
        f'Absolute Error (max: {np.max(error_map):.1f}dB)'
    ]

    for ax, title in zip(axes, titles):
        if add_buildings:
            # Add building overlay
            red_overlay = np.zeros((*osm.shape, 4), dtype=np.uint8)
            red_overlay[osm == 0, 0] = 255
            red_overlay[osm == 0, 3] = 255
            ax.imshow(red_overlay)

        ax.set_title(title, pad=20, fontsize=TITLE_SIZE)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Add colorbar for soundmaps and error maps
    cbar_ax = fig.add_axes([0.87, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='vertical')
    cbar.set_label('decibels (dB)', fontsize=COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    # Add colorbar for absolute error
    cbar_ax2 = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    cbar2 = fig.colorbar(im5, cax=cbar_ax2, orientation='vertical')
    cbar2.set_label('Absolute Error (dB)', fontsize=COLORBAR_LABEL_SIZE)
    cbar2.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    return fig

def find_sample_by_name(test_csv, sample_name):
    """Find a sample in the test CSV by its soundmap name."""
    sample = test_csv[test_csv['soundmap'].str.contains(sample_name, na=False)]
    if len(sample) == 0:
        raise ValueError(f"Sample {sample_name} not found in the test dataset")
    return sample.iloc[0]

def main():
    parser = argparse.ArgumentParser(description='Visualize soundmap predictions with error analysis')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the test data directory')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Path to the predictions directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save the visualization results')
    parser.add_argument('--sample_name', type=str,
                        help='Specific sample name to visualize (e.g., 23773_LEQ_256)')
    parser.add_argument('--sample_index', type=int,
                        help='Specific sample index to visualize')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of random samples to visualize (if no specific sample is specified)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data info
    test_csv = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    samples_to_process = []

    if args.sample_name:
        # Find specific sample by name
        sample = find_sample_by_name(test_csv, args.sample_name)
        samples_to_process.append(sample)
    elif args.sample_index is not None:
        # Use specific index
        samples_to_process.append(test_csv.iloc[args.sample_index])
    else:
        # Random samples
        samples_to_process = test_csv.sample(n=args.num_samples)

    # Process each sample
    for sample in samples_to_process:
        # Plot with buildings
        fig_with_buildings = plot_soundmaps(sample, args.pred_dir, args.data_dir, add_buildings=True)
        sample_name = os.path.splitext(os.path.basename(sample.soundmap))[0]
        output_path = os.path.join(args.output_dir, f"{sample_name}_comparison_with_buildings.png")
        fig_with_buildings.savefig(output_path, bbox_inches='tight', pad_inches=0.2, dpi=300)
        plt.close(fig_with_buildings)
        print(f"Visualization with buildings saved to: {output_path}")

        # Plot without buildings
        fig_without_buildings = plot_soundmaps(sample, args.pred_dir, args.data_dir, add_buildings=False)
        output_path = os.path.join(args.output_dir, f"{sample_name}_comparison_without_buildings.png")
        fig_without_buildings.savefig(output_path, bbox_inches='tight', pad_inches=0.2, dpi=300)
        plt.close(fig_without_buildings)
        print(f"Visualization without buildings saved to: {output_path}")

if __name__ == "__main__":
    main()