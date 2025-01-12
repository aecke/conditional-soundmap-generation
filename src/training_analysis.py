import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd

def load_training_logs(log_dir):
    """Load all training log files and combine their data."""
    all_entries = []
    log_files = [f for f in os.listdir(log_dir) if f.startswith('training_metrics_')]
    
    print(f"\nFound {len(log_files)} log files")
    
    for log_file in sorted(log_files):
        print(f"\nProcessing {log_file}...")
        entries_in_file = 0
        
        with open(os.path.join(log_dir, log_file), 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    
                    # Debug first entry structure
                    if line_num == 1:
                        print("\nFirst entry structure:")
                        print(json.dumps(entry, indent=2))
                    
                    metrics = {
                        'iteration': entry['iteration'],
                        'timestamp': entry['timestamp']
                    }
                    
                    # Extract training metrics
                    train_metrics = entry.get('train_metrics', {})
                    if train_metrics:
                        metrics.update({
                            'loss': float(train_metrics.get('loss')),
                            'loss_left': float(train_metrics.get('loss_left', 0)),
                            'loss_right': float(train_metrics.get('loss_right', 0)),
                            'learning_rate': float(train_metrics.get('learning_rate', 0))
                        })
                    
                    # Extract validation metrics if present
                    val_metrics = entry.get('validation_metrics')
                    if val_metrics:
                        if val_metrics.get('val_loss') is not None:
                            metrics['val_loss'] = float(val_metrics.get('val_loss'))
                        if val_metrics.get('best_val_loss') is not None:
                            metrics['best_val_loss'] = float(val_metrics.get('best_val_loss'))
                    
                    all_entries.append(metrics)
                    entries_in_file += 1
                    
                    # Debug progress
                    if line_num % 1000 == 0:
                        print(f"Processed {line_num} lines...")
                        
                except Exception as e:
                    print(f"Error parsing line {line_num}: {str(e)}")
                    if line_num == 1:
                        print("Problematic line content:")
                        print(line[:500])  # Print first 500 chars of the line
    
        print(f"Parsed {entries_in_file} entries from {log_file}")
    
    # Create DataFrame
    df = pd.DataFrame(all_entries)
    
    # Print DataFrame info
    print("\nDataFrame Info:")
    print(df.info())
    print("\nDataFrame head:")
    print(df.head().to_string())
    print("\nAvailable columns:", df.columns.tolist())
    
    return df

def analyze_loss_relationships(df):
    """Analyze relationships between different loss components."""
    # Verify data availability
    required_columns = ['iteration', 'loss', 'loss_left', 'loss_right']
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame!")
            return None
            
    plt.figure(figsize=(15, 10))
    
    try:
        # Plot 1: Loss Components Over Time
        plt.subplot(221)
        if 'loss' in df.columns:
            plt.plot(df['iteration'], df['loss'], label='Total Loss', alpha=0.7)
        if 'loss_left' in df.columns:
            plt.plot(df['iteration'], df['loss_left'], label='Building Encoder Loss', alpha=0.7)
        if 'loss_right' in df.columns:
            plt.plot(df['iteration'], df['loss_right'], label='Soundmap Generator Loss', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Loss Value')
        plt.title('Loss Components Over Training')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Loss Right vs Loss Left
        plt.subplot(222)
        plt.scatter(df['loss_left'], df['loss_right'], alpha=0.5, s=1)
        plt.xlabel('Building Encoder Loss')
        plt.ylabel('Soundmap Generator Loss')
        plt.title('Generator Loss vs Encoder Loss')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Moving Averages
        window = 100
        plt.subplot(223)
        if 'loss_right' in df.columns:
            plt.plot(df['iteration'], 
                    df['loss_right'].rolling(window=window).mean(), 
                    label=f'Generator Loss ({window}-iter avg)')
        if 'loss_left' in df.columns:
            plt.plot(df['iteration'], 
                    df['loss_left'].rolling(window=window).mean(), 
                    label=f'Encoder Loss ({window}-iter avg)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss Value (Moving Average)')
        plt.title('Smoothed Loss Components')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 4: Loss Ratios
        plt.subplot(224)
        if all(x in df.columns for x in ['loss_right', 'loss_left']):
            ratio = df['loss_right'] / df['loss_left']
            plt.plot(df['iteration'], ratio.rolling(window=window).mean())
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
            plt.xlabel('Iteration')
            plt.ylabel('Generator/Encoder Loss Ratio')
            plt.title('Loss Component Ratio Over Time')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt.gcf()
    
    except Exception as e:
        print(f"Error creating loss plots: {str(e)}")
        return None
    
    plt.tight_layout()
    return plt.gcf()

def analyze_validation_trends(df):
    """Analyze validation loss trends."""
    val_data = df[df['val_loss'].notna()]
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(val_data['iteration'], val_data['val_loss'], label='Validation Loss')
    plt.plot(val_data['iteration'], val_data['best_val_loss'], 
             label='Best Validation Loss', linestyle='--')
    
    plt.xlabel('Iteration')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Trends')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    return plt.gcf()

def print_statistics(df):
    """Print relevant training statistics."""
    print("\nTraining Statistics:")
    print("===================")
    
    # Basic stats
    print(f"\nTotal Iterations: {len(df):,}")
    duration = (pd.to_datetime(df['timestamp'].iloc[-1]) - 
               pd.to_datetime(df['timestamp'].iloc[0])).total_seconds() / 3600
    print(f"Training Duration: {duration:.2f} hours")
    
    # Loss statistics
    print("\nLoss Component Statistics:")
    for component in ['loss', 'loss_left', 'loss_right']:
        data = df[component].dropna()
        print(f"\n{component}:")
        print(f"  Initial: {data.iloc[0]:.4f}")
        print(f"  Final: {data.iloc[-1]:.4f}")
        print(f"  Min: {data.min():.4f}")
        print(f"  Max: {data.max():.4f}")
        print(f"  Mean: {data.mean():.4f}")
        print(f"  Std: {data.std():.4f}")
    
    # Loss ratios
    ratio = df['loss_right'] / df['loss_left']
    print("\nLoss Ratio (Generator/Encoder):")
    print(f"  Initial: {ratio.iloc[0]:.4f}")
    print(f"  Final: {ratio.iloc[-1]:.4f}")
    print(f"  Mean: {ratio.mean():.4f}")
    
    # Validation statistics
    if 'val_loss' in df.columns:
        val_data = df['val_loss'].dropna()
        print("\nValidation Loss:")
        print(f"  Best: {val_data.min():.4f}")
        print(f"  Final: {val_data.iloc[-1]:.4f}")
        print(f"  Mean: {val_data.mean():.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze training logs")
    parser.add_argument("--log_dir", type=str, required=True,
                       help="Directory containing training logs")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and process logs
    print("Loading training logs...")
    df = load_training_logs(args.log_dir)
    
    # Create and save plots
    print("\nCreating analysis plots...")
    loss_fig = analyze_loss_relationships(df)
    loss_fig.savefig(os.path.join(args.output_dir, 'loss_analysis.png'))
    
    if 'val_loss' in df.columns:
        val_fig = analyze_validation_trends(df)
        val_fig.savefig(os.path.join(args.output_dir, 'validation_analysis.png'))
    
    # Print statistics
    print_statistics(df)
    
    print(f"\nAnalysis results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()