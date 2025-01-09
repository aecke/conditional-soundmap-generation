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
import seaborn as sns
from scipy import stats

class SoundMapValidator:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def validate_soundmap(self, noisemap, name="", save_prefix=""):
        """Validate and visualize a soundmap's value distribution."""
        stats = {
            "min": float(np.min(noisemap)),
            "max": float(np.max(noisemap)),
            "mean": float(np.mean(noisemap)),
            "median": float(np.median(noisemap)),
            "std": float(np.std(noisemap)),
            "zeros_percent": float(np.mean(noisemap == 0) * 100),
            "unique_values": int(len(np.unique(noisemap)))
        }
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(noisemap.flatten(), bins=50, density=True, alpha=0.7)
        plt.axvline(stats["mean"], color='r', linestyle='dashed', linewidth=2, label=f'Mean: {stats["mean"]:.2f}')
        plt.axvline(stats["median"], color='g', linestyle='dashed', linewidth=2, label=f'Median: {stats["median"]:.2f}')
        plt.title(f"Distribution of values in {name}")
        plt.xlabel("dB")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, f"{save_prefix}_histogram.png"))
        plt.close()
        
        return stats

    def compare_soundmaps(self, true_map, pred_map, save_prefix=""):
        """Compare two soundmaps and visualize their differences."""
        difference = true_map - pred_map
        
        # Create difference heatmap
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.imshow(true_map, cmap='viridis')
        plt.title("Ground Truth")
        plt.colorbar()
        
        plt.subplot(132)
        plt.imshow(pred_map, cmap='viridis')
        plt.title("Prediction")
        plt.colorbar()
        
        plt.subplot(133)
        plt.imshow(difference, cmap='RdBu')
        plt.clim(vmin=-np.max(np.abs(difference)), vmax=np.max(np.abs(difference)))  # Symmetrische Farbskala
        plt.title("Difference (True - Pred)")
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{save_prefix}_comparison.png"))
        plt.close()
        
        # Create scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(true_map.flatten(), pred_map.flatten(), alpha=0.1, s=1)
        plt.plot([0, 100], [0, 100], 'r--', label='Perfect Prediction')
        plt.xlabel("Ground Truth (dB)")
        plt.ylabel("Prediction (dB)")
        plt.title("Prediction vs Ground Truth")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, f"{save_prefix}_scatter.png"))
        plt.close()

class SoundMapMetrics:
    def __init__(self, min_db_threshold=30):
        self.min_db_threshold = min_db_threshold
    
    def load_soundmap(self, path):
        """Load and normalize soundmap."""
        return (1 - np.array(Image.open(path).convert("L"), dtype=np.float32) / 255) * 100
    
    def calc_basic_metrics(self, true_map, pred_map):
        """Calculate basic error metrics."""
        # Standard MAE
        mae = np.mean(np.abs(true_map - pred_map))
        
        # RMSE
        rmse = np.sqrt(np.mean((true_map - pred_map) ** 2))
        
        # Maximum Error
        max_error = np.max(np.abs(true_map - pred_map))
        
        # Percentage within different error thresholds
        within_1db = np.mean(np.abs(true_map - pred_map) < 1.0) * 100
        within_3db = np.mean(np.abs(true_map - pred_map) < 3.0) * 100
        within_5db = np.mean(np.abs(true_map - pred_map) < 5.0) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'max_error': max_error,
            'within_1db_percent': within_1db,
            'within_3db_percent': within_3db,
            'within_5db_percent': within_5db
        }
    
    def calc_improved_mape(self, true_map, pred_map):
        """Calculate improved MAPE with thresholding."""
        # Only consider values above threshold
        mask = true_map >= self.min_db_threshold
        
        if not np.any(mask):
            return np.nan
        
        # Calculate MAPE for masked values
        error = np.abs((true_map[mask] - pred_map[mask]) / true_map[mask]) * 100
        
        # Remove extreme outliers (> 3 std from mean)
        error_mean = np.mean(error)
        error_std = np.std(error)
        valid_error = error[np.abs(error - error_mean) <= 3 * error_std]
        
        return np.mean(valid_error)
    
    def calc_intensity_based_metrics(self, true_map, pred_map):
        """Calculate metrics for different intensity ranges with safe division."""
        ranges = [
            (0, 40, "low_intensity"),
            (40, 70, "medium_intensity"),
            (70, 100, "high_intensity")
        ]
        
        metrics = {}
        for min_val, max_val, name in ranges:
            mask = (true_map >= min_val) & (true_map < max_val)
            if np.any(mask):
                # MAE calculation
                metrics[f"mae_{name}"] = np.mean(np.abs(true_map[mask] - pred_map[mask]))
                
                # Safe MAPE calculation
                valid_true = true_map[mask]
                valid_pred = pred_map[mask]
                valid_mask = valid_true > 0  # Avoid division by zero
                
                if np.any(valid_mask):
                    mape = np.mean(np.abs((valid_true[valid_mask] - valid_pred[valid_mask]) / 
                                        valid_true[valid_mask])) * 100
                    metrics[f"mape_{name}"] = mape
                else:
                    metrics[f"mape_{name}"] = np.nan
            else:
                metrics[f"mae_{name}"] = np.nan
                metrics[f"mape_{name}"] = np.nan
                
        return metrics

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

def compute_visibility(osm_path, image_size=256):
    """Compute visibility maps with improved error handling."""
    try:
        image_map = np.array(Image.open(osm_path).convert('L').resize((image_size, image_size)))
        image_map = np.where(image_map > 0, 1, 0)
        visibility_map = ray_tracing(image_size, image_map)
        
        pixels_in_sight = np.logical_and(visibility_map == 1, image_map == 1)
        pixels_not_in_sight = np.logical_and(visibility_map == 0, image_map == 1)
        
        # Clean up masks
        pixels_not_in_sight = np.where(image_map == 0, 0, pixels_not_in_sight)
        pixels_in_sight = np.where(image_map == 0, 0, pixels_in_sight)
        
        return pixels_in_sight, pixels_not_in_sight
    except Exception as e:
        print(f"Error in visibility computation: {str(e)}")
        return None, None

def evaluate_sample(metrics, validator, true_path, pred_path, osm_path=None, save_prefix=None):
    """Evaluate a single sample with comprehensive metrics and validation."""
    # Load maps
    true_map = metrics.load_soundmap(true_path)
    pred_map = metrics.load_soundmap(pred_path)
    
    # Validate soundmaps
    if save_prefix:
        true_stats = validator.validate_soundmap(true_map, "Ground Truth", save_prefix)
        pred_stats = validator.validate_soundmap(pred_map, "Prediction", save_prefix)
        validator.compare_soundmaps(true_map, pred_map, save_prefix)
    
    # Calculate basic metrics
    basic_metrics = metrics.calc_basic_metrics(true_map, pred_map)
    
    # Calculate improved MAPE
    mape = metrics.calc_improved_mape(true_map, pred_map)
    
    # Calculate intensity-based metrics
    intensity_metrics = metrics.calc_intensity_based_metrics(true_map, pred_map)
    
    # Combine all metrics
    all_metrics = {
        **basic_metrics,
        'mape': mape,
        **intensity_metrics
    }
    
    # Calculate line-of-sight metrics if building map is provided
    if osm_path is not None:
        pixels_in_sight, pixels_not_in_sight = compute_visibility(osm_path)
        if pixels_in_sight is not None:
            # Create masked arrays for LoS evaluation
            true_in_sight = np.where(pixels_in_sight, true_map, np.nan)
            pred_in_sight = np.where(pixels_in_sight, pred_map, np.nan)
            true_not_in_sight = np.where(pixels_not_in_sight, true_map, np.nan)
            pred_not_in_sight = np.where(pixels_not_in_sight, pred_map, np.nan)
            
            # Calculate LoS metrics
            all_metrics.update({
                'los_mae': np.nanmean(np.abs(true_in_sight - pred_in_sight)),
                'nlos_mae': np.nanmean(np.abs(true_not_in_sight - pred_not_in_sight)),
                'los_mape': metrics.calc_improved_mape(true_in_sight[~np.isnan(true_in_sight)], 
                                                     pred_in_sight[~np.isnan(pred_in_sight)]),
                'nlos_mape': metrics.calc_improved_mape(true_not_in_sight[~np.isnan(true_not_in_sight)], 
                                                      pred_not_in_sight[~np.isnan(pred_not_in_sight)])
            })
    
    return all_metrics

def create_summary_plots(results_df, plot_dir):
    """Create comprehensive summary plots."""
    # Set up the style
    # Use default style with grid
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    
    # Create correlation matrix
    plt.figure(figsize=(12, 10))
    correlation = results_df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation, annot=True, cmap='RdBu', center=0)
    plt.title('Metric Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'metric_correlations.png'))
    plt.close()
    
    # Create box plots for all metrics
    plt.figure(figsize=(15, 8))
    results_df.select_dtypes(include=[np.number]).boxplot()
    plt.xticks(rotation=45)
    plt.title('Distribution of Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'metric_distributions.png'))
    plt.close()
    
    # Create violin plots for intensity-based metrics
    intensity_metrics = [col for col in results_df.columns if any(x in col for x in ['low_intensity', 'medium_intensity', 'high_intensity'])]
    if intensity_metrics:
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=results_df[intensity_metrics])
        plt.xticks(rotation=45)
        plt.title('Distribution of Intensity-Based Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'intensity_distributions.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Improved evaluation of generated soundmaps")
    parser.add_argument("--data_dir", type=str, required=True,
                      help="Directory containing test dataset")
    parser.add_argument("--pred_dir", type=str, required=True,
                      help="Directory containing predictions to evaluate")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory for evaluation outputs")
    parser.add_argument("--model_type", type=str, required=True,
                      help="Type of model being evaluated")
    parser.add_argument("--min_db_threshold", type=float, default=30,
                      help="Minimum dB threshold for MAPE calculation")
    args = parser.parse_args()
    
    # Setup directories
    eval_dir = os.path.join(args.output_dir, args.model_type, 'metrics')
    os.makedirs(eval_dir, exist_ok=True)
    
    validation_dir = os.path.join(eval_dir, 'validations')
    os.makedirs(validation_dir, exist_ok=True)
    
    # Initialize metrics and validator
    metrics = SoundMapMetrics(min_db_threshold=args.min_db_threshold)
    validator = SoundMapValidator(validation_dir)
    
    # Load test dataset
    test_csv = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    results = []
    errors = []

    print(f"\nStarting improved metric evaluation for {len(test_csv)} samples...")
    
    # Process each sample
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

            # Run detailed evaluation for first 5 samples
            save_prefix = f"sample_{index}" if index < 5 else None
            
            # Calculate metrics
            sample_metrics = evaluate_sample(
                metrics,
                validator,
                true_soundmap_path,
                pred_path,
                building_path,
                save_prefix
            )
            
            # Add sample ID and append results
            sample_metrics['sample_id'] = sample_row.sample_id
            results.append(sample_metrics)
            
        except Exception as e:
            error_msg = f"Error processing sample {index}: {str(e)}"
            errors.append(error_msg)
            print(f"\n{error_msg}")
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary_stats = results_df.select_dtypes(include=[np.number]).describe()
    
    # Create additional statistical analysis
    detailed_analysis = {
        'metric_correlations': results_df.select_dtypes(include=[np.number]).corr().to_dict(),
        'metric_distributions': {
            col: {
                'skewness': float(stats.skew(results_df[col].dropna())),
                'kurtosis': float(stats.kurtosis(results_df[col].dropna())),
                'normality_test': float(stats.normaltest(results_df[col].dropna())[1])
            }
            for col in results_df.select_dtypes(include=[np.number]).columns
        }
    }
    
    # Create visualization plots
    create_summary_plots(results_df, validation_dir)
    
    # Save results
    results_df.to_csv(os.path.join(eval_dir, 'detailed_results.csv'), index=False)
    summary_stats.to_csv(os.path.join(eval_dir, 'summary_statistics.csv'))
    
    with open(os.path.join(eval_dir, 'detailed_analysis.json'), 'w') as f:
        json.dump(detailed_analysis, f, indent=4)
    
    if errors:
        error_log_path = os.path.join(eval_dir, 'evaluation_errors.txt')
        with open(error_log_path, 'w') as f:
            f.write('\n'.join(errors))

    # Print summary report
    print("\nEvaluation Results Summary:")
    print("\nBasic Metrics:")
    basic_metrics = ['mae', 'rmse', 'mape']
    for metric in basic_metrics:
        if metric in results_df.columns:
            values = results_df[metric].dropna()
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {values.mean():.4f}")
            print(f"  Median: {values.median():.4f}")
            print(f"  Std: {values.std():.4f}")
            print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")
    
    print("\nIntensity-Based Analysis:")
    intensity_metrics = [col for col in results_df.columns if any(x in col for x in ['low_intensity', 'medium_intensity', 'high_intensity'])]
    for metric in intensity_metrics:
        values = results_df[metric].dropna()
        print(f"\n{metric}:")
        print(f"  Mean: {values.mean():.4f}")
        print(f"  Median: {values.median():.4f}")
    
    print("\nLine of Sight Analysis:")
    los_metrics = ['los_mae', 'nlos_mae', 'los_mape', 'nlos_mape']
    for metric in los_metrics:
        if metric in results_df.columns:
            values = results_df[metric].dropna()
            print(f"\n{metric}:")
            print(f"  Mean: {values.mean():.4f}")
            print(f"  Median: {values.median():.4f}")
    
    print("\nAccuracy Thresholds:")
    threshold_metrics = ['within_1db_percent', 'within_3db_percent', 'within_5db_percent']
    for metric in threshold_metrics:
        if metric in results_df.columns:
            values = results_df[metric].dropna()
            print(f"\n{metric}:")
            print(f"  Mean: {values.mean():.2f}%")
    
    print(f"\nDetailed results saved to: {eval_dir}")
    print(f"Validation visualizations saved to: {validation_dir}")

if __name__ == "__main__":
    main()