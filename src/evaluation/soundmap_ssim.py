import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json

class SoundMapSSIMAnalyzer:
    def __init__(self, data_dir, pred_dir, output_dir, model_type):
        """Initialize the SSIM analyzer with required paths."""
        self.data_dir = data_dir
        self.pred_dir = pred_dir
        self.output_dir = os.path.join(output_dir, model_type, 'ssim_analysis')
        self.model_type = model_type
        os.makedirs(self.output_dir, exist_ok=True)

    def normalize_image(self, image_array):
        """Convert image to dB scale and normalize."""
        db_values = (1 - image_array / 255) * 100
        return db_values

    def calculate_ssim(self, true_path, pred_path):
        """Calculate SSIM between true and predicted soundmaps."""
        # Load images
        true_img = np.array(Image.open(true_path).convert("L"))
        pred_img = np.array(Image.open(pred_path).convert("L"))
        
        # Convert to dB scale
        true_db = self.normalize_image(true_img)
        pred_db = self.normalize_image(pred_img)
        
        # Calculate SSIM
        ssim_value = ssim(true_db, pred_db, data_range=100.0)
        return ssim_value

    def create_visualizations(self, ssim_values, detailed_results):
        """Create visualization plots for SSIM analysis."""
        # Distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(ssim_values, bins=30, kde=True)
        plt.title(f'SSIM Distribution - {self.model_type.title()} Model')
        plt.xlabel('SSIM Value')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.output_dir, 'ssim_distribution.png'))
        plt.close()

        # Example visualizations for best and worst cases
        self.visualize_examples(detailed_results)

    def visualize_examples(self, results):
        """Create visualizations for best and worst SSIM cases."""
        sorted_results = sorted(results, key=lambda x: x['ssim'])
        worst_case = sorted_results[:1]
        best_case = sorted_results[-1:]
        cases = best_case + worst_case

        # --- Einstellungen für wissenschaftliche Publikationen ---
        plt.rcParams.update({'font.size': 10,
                             'axes.titlesize': 10,
                             'axes.labelsize': 9,
                             'xtick.labelsize': 9,
                             'ytick.labelsize': 9,
                             'legend.fontsize': 9,
                             'figure.dpi': 300
                            })

        fig, axes = plt.subplots(2, 3, figsize=(8.27, 5))
        fig.suptitle(f'{self.model_type.title()} - Best & Worst SSIM Cases', fontsize=12, y=0.96) # y verringert

        for i, case in enumerate(cases):
            # Load images
            true_img = np.array(Image.open(case['true_path']).convert("L"))
            pred_img = np.array(Image.open(case['pred_path']).convert("L"))

            # Convert to dB scale
            true_db = self.normalize_image(true_img)
            pred_db = self.normalize_image(pred_img)
            diff = np.abs(true_db - pred_db)

            # Fix color scale to ground truth for each row
            vmin = np.min(true_db)
            vmax = np.max(true_db)

            # --- Ground Truth ---
            im1 = axes[i, 0].imshow(true_db, cmap='viridis', vmin=vmin, vmax=vmax)
            if i == 0:
                axes[i, 0].set_title(f'Ground Truth\nBest Case (Sample: {case["sample_id"]})')
            else:
                axes[i, 0].set_title(f'Ground Truth\nWorst Case (Sample: {case["sample_id"]})')
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
            axes[i, 0].set_aspect('equal')

            # --- Prediction ---
            im2 = axes[i, 1].imshow(pred_db, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f'Prediction\nSSIM: {case["ssim"]:.4f}')
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
            axes[i, 1].set_aspect('equal')

            # Colorbar for Prediction (right of Prediction, same height as image)
            cbar1 = fig.colorbar(im2, ax=axes[i, 1], label='dB', fraction=0.046, pad=0.04)

            # --- Absolute Difference ---
            im3 = axes[i, 2].imshow(diff, cmap='RdYlGn_r')
            axes[i, 2].set_title('Abs. Difference')
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
            axes[i, 2].set_aspect('equal')

            # Colorbar for Absolute Difference (right of Difference, same height as image)
            cbar2 = fig.colorbar(im3, ax=axes[i, 2], label='dB Diff.', fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.94]) # rect angepasst
        plt.subplots_adjust(wspace=0.1, hspace=0.4)
        plt.savefig(os.path.join(self.output_dir, 'best_worst_cases.png'))
        plt.close()

    def analyze(self):
        """Perform SSIM analysis on the prediction dataset."""
        print(f"\nPerforming SSIM analysis for {self.model_type} model...")
        
        # Load test dataset info
        test_csv = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        results = []
        ssim_values = []
        
        # Process each sample
        for idx, row in tqdm(test_csv.iterrows(), total=len(test_csv), desc="Calculating SSIM"):
            try:
                # Construct paths
                pred_path = os.path.join(self.pred_dir, f"y_{idx}.png")
                true_path = os.path.join(self.data_dir, row.soundmap.replace("./", ""))
                
                # Calculate SSIM
                ssim_value = self.calculate_ssim(true_path, pred_path)
                ssim_values.append(ssim_value)
                
                # Store detailed results
                results.append({
                    'sample_id': idx,
                    'ssim': ssim_value,
                    'true_path': true_path,
                    'pred_path': pred_path
                })
                
            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
                continue
        
        # Calculate statistics
        stats = {
            'mean_ssim': float(np.mean(ssim_values)),
            'median_ssim': float(np.median(ssim_values)),
            'std_ssim': float(np.std(ssim_values)),
            'min_ssim': float(np.min(ssim_values)),
            'max_ssim': float(np.max(ssim_values)),
            'num_samples': len(ssim_values)
        }
        
        # Create visualizations
        self.create_visualizations(ssim_values, results)
        
        # Save results
        with open(os.path.join(self.output_dir, 'ssim_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=4)
            
        pd.DataFrame(results).to_csv(
            os.path.join(self.output_dir, 'detailed_results.csv'), 
            index=False
        )
        
        # Print summary
        print("\nSSIM Analysis Results:")
        print(f"Number of samples analyzed: {stats['num_samples']}")
        print(f"Mean SSIM: {stats['mean_ssim']:.4f}")
        print(f"Median SSIM: {stats['median_ssim']:.4f}")
        print(f"Std Dev SSIM: {stats['std_ssim']:.4f}")
        print(f"Range: [{stats['min_ssim']:.4f}, {stats['max_ssim']:.4f}]")
        print(f"\nResults saved to: {self.output_dir}")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Calculate SSIM for soundmap predictions")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing test dataset")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing predictions to evaluate")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for evaluation outputs")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Type of model being evaluated (baseline/reflection/diffraction/combined)")
    
    args = parser.parse_args()
    
    analyzer = SoundMapSSIMAnalyzer(
        args.data_dir,
        args.pred_dir,
        args.output_dir,
        args.model_type
    )
    
    analyzer.analyze()

if __name__ == "__main__":
    main()