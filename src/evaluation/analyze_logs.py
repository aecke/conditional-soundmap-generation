import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class LogAnalyzer:
    def __init__(self, log_dir, output_dir):
        self.log_dir = log_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def parse_log_file(self, log_path):
        """Parse a single log file."""
        data = []
        with open(log_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    # Extract metrics
                    metrics = entry.get('metrics', {})
                    record = {
                        'iteration': entry.get('iteration'),
                        'time': entry.get('time_taken'),
                        'loss': metrics.get('loss'),
                        'loss_left': metrics.get('loss_left'),
                        'loss_right': metrics.get('loss_right'),
                        'val_loss': metrics.get('val_loss'),
                        'learning_rate': metrics.get('learning_rate')
                    }
                    data.append(record)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line {line_num} in {os.path.basename(log_path)}")
                except Exception as e:
                    print(f"Error processing line {line_num} in {os.path.basename(log_path)}: {str(e)}")
                    
        return pd.DataFrame(data)
    
    def analyze_logs(self):
        """Analyze all log files in directory."""
        log_files = glob.glob(os.path.join(self.log_dir, "training_log_*.json"))
        print(f"Found {len(log_files)} log files")
        
        all_data = []
        for log_file in log_files:
            print(f"\nProcessing {os.path.basename(log_file)}...")
            df = self.parse_log_file(log_file)
            all_data.append(df)
            
        if not all_data:
            print("No data found in log files!")
            return
            
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('iteration').reset_index(drop=True)
        
        # Save raw data
        combined_df.to_csv(os.path.join(self.output_dir, 'training_metrics.csv'), index=False)
        
        # Create plots
        self.plot_training_curves(combined_df)
        
        # Print statistics
        self.print_statistics(combined_df)
    
    def plot_training_curves(self, df):
        """Create training curve plots."""
        # Loss curves
        plt.figure(figsize=(12, 8))
        if 'loss' in df:
            plt.plot(df['iteration'], df['loss'], label='Total Loss', alpha=0.7)
        if 'loss_left' in df:
            plt.plot(df['iteration'], df['loss_left'], label='Left Loss', alpha=0.7)
        if 'loss_right' in df:
            plt.plot(df['iteration'], df['loss_right'], label='Right Loss', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'training_losses.png'))
        plt.close()
        
        # Validation loss
        if 'val_loss' in df:
            plt.figure(figsize=(10, 6))
            valid_val = df[df['val_loss'].notna()]
            plt.plot(valid_val['iteration'], valid_val['val_loss'])
            plt.xlabel('Iteration')
            plt.ylabel('Validation Loss')
            plt.title('Validation Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'validation_loss.png'))
            plt.close()
        
        # Learning rate
        if 'learning_rate' in df:
            plt.figure(figsize=(10, 6))
            valid_lr = df[df['learning_rate'].notna()]
            plt.plot(valid_lr['iteration'], valid_lr['learning_rate'])
            plt.xlabel('Iteration')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'learning_rate.png'))
            plt.close()
    
    def print_statistics(self, df):
        """Print training statistics."""
        stats = {
            'Training Duration (hours)': (df['time'].max() - df['time'].min()) / 3600,
            'Total Iterations': len(df),
            'Final Loss': df['loss'].iloc[-1],
            'Best Loss': df['loss'].min(),
            'Mean Loss': df['loss'].mean(),
            'Loss Std Dev': df['loss'].std()
        }
        
        if 'val_loss' in df:
            stats.update({
                'Final Val Loss': df['val_loss'].iloc[-1],
                'Best Val Loss': df['val_loss'].min(),
                'Mean Val Loss': df['val_loss'].mean(),
                'Val Loss Std Dev': df['val_loss'].std()
            })
        
        # Save statistics
        with open(os.path.join(self.output_dir, 'training_statistics.txt'), 'w') as f:
            f.write("Training Statistics\n")
            f.write("===================\n\n")
            for key, value in stats.items():
                f.write(f"{key}: {value:.4f}\n")
        
        # Print to console
        print("\nTraining Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze training logs")
    parser.add_argument("--log_dir", type=str, required=True,
                       help="Directory containing training logs")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    analyzer = LogAnalyzer(args.log_dir, args.output_dir)
    analyzer.analyze_logs()

if __name__ == "__main__":
    main()