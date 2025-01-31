import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import seaborn as sns

class SoundMapValidator:
    def __init__(self, data_dir, pred_dir, output_dir):
        """
        Initialize validator with directory paths.
        
        Args:
            data_dir: Directory containing ground truth data
            pred_dir: Directory containing predictions
            output_dir: Directory to save validation results
        """
        self.data_dir = data_dir
        self.pred_dir = pred_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_and_normalize_image(self, path):
        """Load image and convert to dB scale."""
        img = np.array(Image.open(path).convert("L"), dtype=np.float32)
        db_values = (1 - img / 255) * 100
        return img, db_values
    
    def validate_image_values(self, image_path, title):
        """Validate raw and normalized values of an image."""
        raw_img, db_values = self.load_and_normalize_image(image_path)
        
        stats = {
            'raw_min': raw_img.min(),
            'raw_max': raw_img.max(),
            'raw_mean': raw_img.mean(),
            'db_min': db_values.min(),
            'db_max': db_values.max(),
            'db_mean': db_values.mean(),
            'unique_values': len(np.unique(db_values))
        }
        
        print(f"\nValidation for {title}:")
        print(f"Raw pixel range: {stats['raw_min']}-{stats['raw_max']} (mean: {stats['raw_mean']:.2f})")
        print(f"dB value range: {stats['db_min']:.2f}-{stats['db_max']:.2f} (mean: {stats['db_mean']:.2f})")
        print(f"Unique values: {stats['unique_values']}")
        
        return stats, db_values
    
    def plot_value_distributions(self, true_path, pred_path, index):
        """Plot value distributions for both true and predicted images."""
        _, true_db = self.load_and_normalize_image(true_path)
        _, pred_db = self.load_and_normalize_image(pred_path)
        
        plt.figure(figsize=(15, 5))
        
        # Ground Truth distribution
        plt.subplot(121)
        plt.hist(true_db.flatten(), bins=50, alpha=0.7, label='Ground Truth')
        plt.axvline(true_db.mean(), color='r', linestyle='dashed', label=f'Mean: {true_db.mean():.2f}')
        plt.title('Ground Truth dB Distribution')
        plt.xlabel('dB')
        plt.ylabel('Count')
        plt.legend()
        
        # Prediction distribution
        plt.subplot(122)
        plt.hist(pred_db.flatten(), bins=50, alpha=0.7, label='Prediction')
        plt.axvline(pred_db.mean(), color='r', linestyle='dashed', label=f'Mean: {pred_db.mean():.2f}')
        plt.title('Prediction dB Distribution')
        plt.xlabel('dB')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'distribution_sample_{index}.png'))
        plt.close()
        
        return true_db, pred_db
    
    def create_error_heatmap(self, true_db, pred_db, index):
        """Create and save error heatmap."""
        error = np.abs(true_db - pred_db)
        
        plt.figure(figsize=(15, 5))
        
        # Original images
        plt.subplot(131)
        plt.imshow(true_db, cmap='viridis')
        plt.colorbar(label='dB')
        plt.title('Ground Truth')
        
        plt.subplot(132)
        plt.imshow(pred_db, cmap='viridis')
        plt.colorbar(label='dB')
        plt.title('Prediction')
        
        # Error heatmap
        plt.subplot(133)
        plt.imshow(error, cmap='RdYlGn_r')  # _r für reverse - rot für hohe Fehler, grün für niedrige
        plt.clim(0, np.max(error))  # Setze Farbskala von 0 bis max error
        cbar = plt.colorbar(label='|Error| (dB)')
        plt.title(f'Absolute Error (Max: {np.max(error):.1f} dB)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'error_heatmap_sample_{index}.png'))
        plt.close()
        
        return error
    
    def analyze_error_patterns(self, true_db, pred_db):
        """Analyze error patterns and relationships."""
        error = np.abs(true_db - pred_db)
        
        # Error vs. True Value relationship
        plt.figure(figsize=(10, 6))
        plt.hexbin(true_db.flatten(), error.flatten(), gridsize=30, cmap='YlOrRd')
        plt.colorbar(label='Count')
        plt.xlabel('True dB Value')
        plt.ylabel('Absolute Error')
        plt.title('Error vs. True Value Relationship')
        plt.savefig(os.path.join(self.output_dir, 'error_vs_true_value.png'))
        plt.close()
        
        # Error statistics for different dB ranges
        ranges = [(0, 30), (30, 60), (60, 100)]
        range_stats = []
        
        for min_db, max_db in ranges:
            mask = (true_db >= min_db) & (true_db < max_db)
            if np.any(mask):
                range_error = error[mask]
                stats = {
                    'range': f'{min_db}-{max_db}dB',
                    'mean_error': range_error.mean(),
                    'median_error': np.median(range_error),
                    'std_error': range_error.std(),
                    'max_error': range_error.max(),
                    'pixels': np.sum(mask)
                }
                range_stats.append(stats)
        
        return range_stats
    
    def validate_samples(self, num_samples=5):
        """Validate a specified number of samples."""
        test_csv = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        
        all_stats = []
        range_stats_list = []
        
        for index in tqdm(range(min(num_samples, len(test_csv))), desc="Validating samples"):
            sample_row = test_csv.iloc[index]
            
            # Construct paths
            true_path = os.path.join(self.data_dir, sample_row.soundmap.replace("./", ""))
            pred_path = os.path.join(self.pred_dir, f"y_{index}.png")
            
            # Basic value validation
            true_stats, true_db = self.validate_image_values(true_path, f"Ground Truth {index}")
            pred_stats, pred_db = self.validate_image_values(pred_path, f"Prediction {index}")
            
            # Plot distributions
            self.plot_value_distributions(true_path, pred_path, index)
            
            # Create error heatmap
            error = self.create_error_heatmap(true_db, pred_db, index)
            
            # Analyze error patterns
            range_stats = self.analyze_error_patterns(true_db, pred_db)
            
            # Combine stats
            sample_stats = {
                'sample_id': index,
                'true_stats': true_stats,
                'pred_stats': pred_stats,
                'mean_error': error.mean(),
                'median_error': np.median(error),
                'max_error': error.max()
            }
            
            all_stats.append(sample_stats)
            range_stats_list.extend(range_stats)
        
        # Save summary statistics
        self.save_summary(all_stats, range_stats_list)
        
    def convert_to_serializable(self, obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return self.convert_to_serializable(obj.tolist())
        else:
            return obj

    def save_summary(self, all_stats, range_stats):
        """Save summary statistics to a file."""
        # Berechne Statistiken
        summary = {
            'overall_statistics': {
                'samples_analyzed': len(all_stats),
                'average_mean_error': float(np.mean([s['mean_error'] for s in all_stats])),
                'average_median_error': float(np.mean([s['median_error'] for s in all_stats])),
                'max_error_observed': float(max([s['max_error'] for s in all_stats]))
            },
            'value_ranges': {
                'ground_truth': {
                    'min_db': float(min(s['true_stats']['db_min'] for s in all_stats)),
                    'max_db': float(max(s['true_stats']['db_max'] for s in all_stats)),
                    'mean_db': float(np.mean([s['true_stats']['db_mean'] for s in all_stats]))
                },
                'predictions': {
                    'min_db': float(min(s['pred_stats']['db_min'] for s in all_stats)),
                    'max_db': float(max(s['pred_stats']['db_max'] for s in all_stats)),
                    'mean_db': float(np.mean([s['pred_stats']['db_mean'] for s in all_stats]))
                }
            },
            'range_statistics': self.convert_to_serializable(
                pd.DataFrame(range_stats).groupby('range').mean().to_dict()
            ),
            'sample_details': self.convert_to_serializable(all_stats)
        }
        
        # Save as JSON
        import json
        with open(os.path.join(self.output_dir, 'validation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4, sort_keys=True)
            
        print("\nValidation Summary:")
        print("\nOverall Statistics:")
        for key, value in summary['overall_statistics'].items():
            print(f"  {key}: {value}")
            
        print("\nValue Ranges:")
        print("  Ground Truth:")
        for key, value in summary['value_ranges']['ground_truth'].items():
            print(f"    {key}: {value}")
        print("  Predictions:")
        for key, value in summary['value_ranges']['predictions'].items():
            print(f"    {key}: {value}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate soundmap predictions")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing ground truth data")
    parser.add_argument("--pred_dir", type=str, required=True,
                       help="Directory containing predictions")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save validation results")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples to validate")
    
    args = parser.parse_args()
    
    validator = SoundMapValidator(args.data_dir, args.pred_dir, args.output_dir)
    validator.validate_samples(args.num_samples)

if __name__ == "__main__":
    main()