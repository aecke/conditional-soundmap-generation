import torch
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

class CheckpointAnalyzer:
    def __init__(self, checkpoint_dir, output_dir):
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def analyze_sample_generation(self, samples_dir):
        """Analyze generated samples from checkpoints."""
        print("Analyzing generated samples...")
        
        # Find all checkpoint sample directories
        sample_dirs = sorted(glob.glob(os.path.join(samples_dir, "*")))
        if not sample_dirs:
            print("No sample directories found!")
            return
            
        all_stats = []
        for sample_dir in tqdm(sample_dirs):
            iter_num = int(os.path.basename(sample_dir))
            
            # Find all PNG files in directory
            sample_files = glob.glob(os.path.join(sample_dir, "*.png"))
            
            for sample_file in sample_files:
                # Load and analyze image
                img = np.array(Image.open(sample_file).convert("L"))
                db_values = (1 - img / 255) * 100
                
                stats = {
                    'iteration': iter_num,
                    'min_db': float(db_values.min()),
                    'max_db': float(db_values.max()),
                    'mean_db': float(db_values.mean()),
                    'std_db': float(db_values.std()),
                    'unique_values': len(np.unique(db_values))
                }
                all_stats.append(stats)
        
        # Convert to structured format
        import pandas as pd
        stats_df = pd.DataFrame(all_stats)
        
        # Plot statistics
        self.plot_db_statistics(stats_df)
        
        # Save statistics
        stats_df.to_csv(os.path.join(self.output_dir, 'sample_statistics.csv'), index=False)
        
        # Print summary
        self.print_db_statistics(stats_df)
    
    def plot_db_statistics(self, df):
        """Create plots of dB statistics over training."""
        plt.figure(figsize=(12, 8))
        
        # dB range plot
        plt.fill_between(df['iteration'], df['min_db'], df['max_db'], 
                        alpha=0.3, label='dB Range')
        plt.plot(df['iteration'], df['mean_db'], 'r-', 
                label='Mean dB', linewidth=2)
        
        # Add ±1 std dev
        plt.fill_between(df['iteration'],
                        df['mean_db'] - df['std_db'],
                        df['mean_db'] + df['std_db'],
                        color='r', alpha=0.2, label='±1 Std Dev')
        
        plt.xlabel('Training Iteration')
        plt.ylabel('dB Value')
        plt.title('dB Statistics Over Training')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(self.output_dir, 'db_statistics.png'))
        plt.close()
        
        # Plot number of unique values
        plt.figure(figsize=(10, 6))
        plt.plot(df['iteration'], df['unique_values'])
        plt.xlabel('Training Iteration')
        plt.ylabel('Number of Unique dB Values')
        plt.title('Value Diversity Over Training')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.output_dir, 'value_diversity.png'))
        plt.close()
    
    def print_db_statistics(self, df):
        """Print summary of dB statistics."""
        stats = {
            'Final Mean dB': df['mean_db'].iloc[-1],
            'Final dB Range': f"{df['min_db'].iloc[-1]:.1f} - {df['max_db'].iloc[-1]:.1f}",
            'Overall Mean dB': df['mean_db'].mean(),
            'Mean dB Std Dev': df['mean_db'].std(),
            'Mean Unique Values': df['unique_values'].mean(),
        }
        
        # Save statistics
        with open(os.path.join(self.output_dir, 'db_statistics.txt'), 'w') as f:
            f.write("dB Value Statistics\n")
            f.write("==================\n\n")
            for key, value in stats.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        # Print to console
        print("\ndB Value Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze checkpoints and generated samples")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing checkpoints")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    analyzer = CheckpointAnalyzer(args.checkpoint_dir, args.output_dir)
    
    # Analyze samples in checkpoint directory
    samples_dir = os.path.join(args.checkpoint_dir, 'samples')
    if os.path.exists(samples_dir):
        analyzer.analyze_sample_generation(samples_dir)
    else:
        print(f"No samples directory found at {samples_dir}")

if __name__ == "__main__":
    main()