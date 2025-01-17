import numpy as np
from PIL import Image
import numba
from numba import jit
import argparse
import os
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt
import json

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calc_mae(true_path, pred_path):
    pred_noisemap = (1 - np.array(
        Image.open(pred_path).convert("L"),
        dtype=np.float32
    ) / 255) * 100

    true_noisemap = (1 - np.array(
        Image.open(true_path).convert("L"),
        dtype=np.float32
    ) / 255) * 100
    return MAE(true_noisemap, pred_noisemap)

def calc_mape(true_path, pred_path):
    # Load and process the predicted and true noise maps
    pred_noisemap = (1 - np.array(
        Image.open(pred_path).convert("L"),
        dtype=np.float32
    ) / 255) * 100
    true_noisemap = (1 - np.array(
        Image.open(true_path).convert("L"),
        dtype=np.float32
    ) / 255) * 100

    # Initialize an error map with zeros
    error_map = np.zeros_like(true_noisemap, dtype=np.float32)

    # Find indices where true noisemap is not 0
    nonzero_indices = true_noisemap != 0

    # Calculate percentage error where true noisemap is not 0
    error_map[nonzero_indices] = np.abs((true_noisemap[nonzero_indices] - pred_noisemap[nonzero_indices]) / true_noisemap[nonzero_indices]) * 100

    # For positions where true noisemap is 0 but pred noisemap is not, set error to 100%
    zero_true_indices = (true_noisemap == 0) & (pred_noisemap != 0)
    error_map[zero_true_indices] = 100

    # Calculate the MAPE over the whole image, ignoring positions where both are 0
    return np.mean(error_map)

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
                continue  # Skip the source point itself
            step_dx = dx / steps
            step_dy = dy / steps

            visible = True  # Assume this point is visible unless proven otherwise
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

def compute_visibility(osm_path, image_size=256):
    image_map = np.array(Image.open(osm_path).convert('L').resize((image_size, image_size)))
    image_map = np.where(image_map > 0, 1, 0)
    visibility_map = ray_tracing(image_size, image_map)
    pixels_in_sight = np.logical_and(visibility_map == 1, image_map == 1)
    pixels_not_in_sight = np.logical_and(visibility_map == 0, image_map == 1)
    pixels_not_in_sight = np.where(image_map == 0, 0, pixels_not_in_sight)
    pixels_in_sight = np.where(image_map == 0, 0, pixels_in_sight)
    return pixels_in_sight, pixels_not_in_sight

def masked_mae(true_labels, predictions):
    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    
    # Create a mask where true_labels are not equal to -1
    mask = true_labels != -1
    
    # Filter arrays with the mask
    true_labels = true_labels[mask]
    predictions = predictions[mask]
    
    # Compute the MAE and return
    return MAE(true_labels, predictions)

def masked_mape(true_labels, predictions):
    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    # Create a mask to exclude -1
    mask = true_labels != -1
    
    # Apply the mask to filter arrays
    true_labels_filtered = true_labels[mask]
    predictions_filtered = predictions[mask]
    
    # Initialize an error map with zeros
    error_map = np.zeros_like(true_labels_filtered, dtype=np.float32)

    # Find indices where true noisemap is not 0
    nonzero_indices = true_labels_filtered != 0

    # Calculate percentage error where true noisemap is not 0
    error_map[nonzero_indices] = np.abs((true_labels_filtered[nonzero_indices] - predictions_filtered[nonzero_indices]) / true_labels_filtered[nonzero_indices]) * 100

    # For positions where true noisemap is 0 but pred noisemap is not, set error to 100%
    zero_true_indices = (true_labels_filtered == 0) & (predictions_filtered != 0)
    error_map[zero_true_indices] = 100

    # Calculate the MAPE over the whole image, ignoring positions where both are 0
    return np.mean(error_map)

def calculate_sight_error(true_path, pred_path, osm_path):
    pred_soundmap = (1 - np.array(
        Image.open(pred_path).convert("L"),
        dtype=np.float32
    ) / 255) * 100
    true_soundmap = (1 - np.array(
        Image.open(true_path).convert("L"),
        dtype=np.float32
    ) / 255) * 100
    _, true_pixels_not_in_sight = compute_visibility(osm_path)

    in_sight_soundmap = true_soundmap.copy()
    not_in_sight_soundmap = true_soundmap.copy()
    
    in_sight_pred_soundmap = pred_soundmap.copy()
    not_in_sight_pred_soundmap = pred_soundmap.copy()
    
    #only get the pixels in sight
    for x in range(256):
        for y in range(256):
            if true_pixels_not_in_sight[y, x] == 0:
                not_in_sight_soundmap[y, x] = -1
                not_in_sight_pred_soundmap[y, x] = -1
            else:
                in_sight_soundmap[y, x] = -1
                in_sight_pred_soundmap[y, x] = -1

    return masked_mae(in_sight_soundmap, in_sight_pred_soundmap), masked_mae(not_in_sight_soundmap, not_in_sight_pred_soundmap), masked_mape(in_sight_soundmap, in_sight_pred_soundmap), masked_mape(not_in_sight_soundmap, not_in_sight_pred_soundmap)

def evaluate_sample(true_path, pred_path, osm_path=None):
    mae = calc_mae(true_path, pred_path)
    mape = calc_mape(true_path, pred_path)

    mae_in_sight = mae_not_in_sight = mape_in_sight = mape_not_in_sight = None
    if osm_path is not None:
        mae_in_sight, mae_not_in_sight, mape_in_sight, mape_not_in_sight = calculate_sight_error(
            true_path, pred_path, osm_path
        )
    return mae, mape, mae_in_sight, mae_not_in_sight, mape_in_sight, mape_not_in_sight

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated soundmaps")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing test dataset")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing predictions to evaluate")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for evaluation outputs")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Type of model being evaluated (with_extra_conditions or without_extra_conditions)")
    return parser.parse_args()

def setup_metric_directories(args):
    """Setup directories for metric results."""
    eval_dir = os.path.join(args.output_dir, args.model_type, 'metrics')
    os.makedirs(eval_dir, exist_ok=True)
    
    paths = {
        'detailed_results': os.path.join(eval_dir, 'detailed_results.csv'),
        'summary_stats': os.path.join(eval_dir, 'summary_statistics.csv'),
        'metrics_plots': os.path.join(eval_dir, 'plots'),
    }
    
    os.makedirs(paths['metrics_plots'], exist_ok=True)
    return paths

def create_metric_plots(results_df, plot_dir):
    """Create visualization plots for metrics."""
    metrics = ["MAE", "MAPE", "LoS_MAE", "NLoS_MAE", "LoS_wMAPE", "NLoS_wMAPE"]
    
    # Histogramm für jede Metrik
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.hist(results_df[metric].dropna(), bins=30, edgecolor='black')
        plt.title(f'Distribution of {metric}')
        plt.xlabel(metric)
        plt.ylabel('Count')
        plt.savefig(os.path.join(plot_dir, f'{metric}_distribution.png'))
        plt.close()
    
    # Box-Plot für alle Metriken
    plt.figure(figsize=(12, 6))
    results_df[metrics].boxplot()
    plt.xticks(rotation=45)
    plt.title('Metric Distributions (Box Plot)')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'metrics_boxplot.png'))
    plt.close()

def main():
    args = parse_args()
    
    # Setup output directories
    paths = setup_metric_directories(args)
    
    # Save evaluation config
    eval_info = {
        'model_type': args.model_type,
        'data_dir': args.data_dir,
        'pred_dir': args.pred_dir
    }
    
    with open(os.path.join(os.path.dirname(paths['detailed_results']), 'metric_evaluation_config.json'), 'w') as f:
        json.dump(eval_info, f, indent=4)

    # Load test dataset
    test_csv = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    results = []
    errors = []

    print(f"\nStarting metric evaluation for {len(test_csv)} samples...")
    for index, sample_row in tqdm(test_csv.iterrows(), total=len(test_csv), desc="Evaluating metrics"):
        try:
            # Construct paths
            pred_path = os.path.join(args.pred_dir, f"y_{index}.png")
            true_soundmap_path = os.path.join(args.data_dir, sample_row.soundmap.replace("./", ""))
            building_path = os.path.join(args.data_dir, sample_row.osm.replace("./", ""))

            # Validate paths
            if not all(os.path.exists(p) for p in [pred_path, true_soundmap_path, building_path]):
                missing = [p for p in [pred_path, true_soundmap_path, building_path] if not os.path.exists(p)]
                raise FileNotFoundError(f"Missing files: {missing}")

            # Calculate metrics
            mae, mape, mae_in_sight, mae_not_in_sight, mape_in_sight, mape_not_in_sight = evaluate_sample(
                true_soundmap_path,
                pred_path,
                building_path
            )
            results.append([
                sample_row.sample_id, mae, mape, 
                mae_in_sight, mae_not_in_sight, 
                mape_in_sight, mape_not_in_sight
            ])
            
        except Exception as e:
            error_msg = f"Error processing sample {index}: {str(e)}"
            errors.append(error_msg)
            print(f"\n{error_msg}")
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(
        results, 
        columns=["sample_id", "MAE", "MAPE", "LoS_MAE", "NLoS_MAE", "LoS_wMAPE", "NLoS_wMAPE"]
    )
    
    # Calculate summary statistics
    summary_stats = results_df.describe()
    
    # Create visualization plots
    create_metric_plots(results_df, paths['metrics_plots'])
    
    # Save results
    results_df.to_csv(paths['detailed_results'], index=False)
    summary_stats.to_csv(paths['summary_stats'])
    
    if errors:
        error_log_path = os.path.join(os.path.dirname(paths['detailed_results']), 'metric_evaluation_errors.txt')
        with open(error_log_path, 'w') as f:
            f.write('\n'.join(errors))

    # Print summary statistics
    print("\nMetric Evaluation Results:")
    print("\nSummary Statistics:")
    print(summary_stats)
    
    print("\nDetailed Statistics:")
    for metric in ["MAE", "MAPE", "LoS_MAE", "NLoS_MAE", "LoS_wMAPE", "NLoS_wMAPE"]:
        valid_values = results_df[metric].dropna()
        if len(valid_values) > 0:
            print(f"\n{metric}:")
            print(f"  Mean: {valid_values.mean():.4f}")
            print(f"  Median: {valid_values.median():.4f}")
            print(f"  Std: {valid_values.std():.4f}")
            print(f"  Min: {valid_values.min():.4f}")
            print(f"  Max: {valid_values.max():.4f}")

    print(f"\nEvaluation results saved to: {os.path.dirname(paths['detailed_results'])}")

if __name__ == "__main__":
    main()