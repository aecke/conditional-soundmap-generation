import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import argparse
from datetime import datetime
from matplotlib import rcParams

class ValidationLossAnalyzer:
    def __init__(self, log_dir, output_dir):
        self.log_dir = log_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set scientific plot style
        self.setup_plot_style()

    def setup_plot_style(self):
        """Configure matplotlib for publication-quality plots."""
        import seaborn as sns
        plt.style.use('seaborn-v0_8-paper')

        # Font settings (serifenlose Schrift)
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        rcParams['font.size'] = 11
        rcParams['axes.labelsize'] = 12
        rcParams['axes.titlesize'] = 14
        rcParams['xtick.labelsize'] = 10
        rcParams['ytick.labelsize'] = 10
        rcParams['legend.fontsize'] = 10
        rcParams['figure.titlesize'] = 16

        # Line settings
        rcParams['lines.linewidth'] = 1.5
        rcParams['lines.markersize'] = 6

        # Grid settings
        rcParams['grid.linestyle'] = '--'
        rcParams['grid.alpha'] = 0.2
        rcParams['grid.color'] = 'gray'

        # Figure settings
        rcParams['figure.dpi'] = 300
        rcParams['savefig.dpi'] = 300
        rcParams['figure.figsize'] = [8, 6]
        rcParams['figure.constrained_layout.use'] = True

        # Scientific notation settings
        rcParams['axes.formatter.limits'] = [-3, 3]
        rcParams['axes.formatter.use_mathtext'] = True

    def parse_log_files(self):
        """Parse all log files in directory and extract validation metrics."""
        print("\nParsing training log files...")
        all_data = []

        log_files = sorted(glob.glob(os.path.join(self.log_dir, "training_metrics_*.jsonl")))
        if not log_files:
            raise FileNotFoundError(f"No log files found in {self.log_dir}")

        for log_file in log_files:
            print(f"\nProcessing {os.path.basename(log_file)}...")
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        metrics = {
                            'iteration': entry.get('iteration'),
                            'timestamp': entry.get('timestamp'),
                            'epoch_approx': entry.get('epoch_approx')
                        }

                        val_metrics = entry.get('validation_metrics', {})
                        if val_metrics:
                            metrics.update({
                                'val_loss': val_metrics.get('val_loss'),
                                'best_val_loss': val_metrics.get('best_val_loss'),
                                'steps_without_improvement': val_metrics.get('steps_without_improvement')
                            })

                        train_metrics = entry.get('train_metrics', {})
                        if train_metrics:
                            metrics.update({
                                'train_loss': train_metrics.get('loss'),
                                'learning_rate': train_metrics.get('learning_rate')
                            })

                        extra_cond = entry.get('extra_conditions', {})
                        if extra_cond:
                            metrics.update({
                                'use_extra_cond': extra_cond.get('enabled', False),
                                'active_conditions': ','.join(extra_cond.get('active', []))
                            })

                        all_data.append(metrics)

                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {os.path.basename(log_file)}: {e}")
                        continue

        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def create_scientific_plot(self, val_data, stats, dataset_name, ax_loss):
        """Create publication-quality plots of validation loss."""

        # Main loss plot
        val_line = ax_loss.plot(val_data['iteration'], val_data['val_loss'],
                            label='Validation Loss', color='#1f77b4', linestyle='-', marker='o', markersize=4)
        if 'train_loss' in val_data.columns:
            train_line = ax_loss.plot(val_data['iteration'], val_data['train_loss'],
                                  label='Training Loss', color='#ff7f0e', linestyle='-', marker='x', markersize=4, alpha=0.7)
        
        best_line = ax_loss.axhline(y=stats['best_val_loss'], color='#2ca02c', linestyle='--',
                                label=f'Best Val Loss: {stats["best_val_loss"]:.4f}')

        window = 50
        val_data['val_loss_ma'] = val_data['val_loss'].rolling(window=window).mean()
        ma_line = ax_loss.plot(val_data['iteration'], val_data['val_loss_ma'],
                           label=f'{window}-pt Moving Avg', color='#d62728', linestyle='-.')

        val_std = val_data['val_loss'].rolling(window=window).std()
        ax_loss.fill_between(val_data['iteration'],
                        val_data['val_loss_ma'] - val_std,
                        val_data['val_loss_ma'] + val_std,
                        alpha=0.2, color='#d62728')

        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss Value')
        ax_loss.set_title(f'Validation Loss Analysis - {dataset_name} Dataset')
        ax_loss.grid(True, alpha=0.2, color='gray')
        ax_loss.legend(loc='upper right', frameon=False, ncol=1, fancybox=False, shadow=False)

        # Add annotations with key statistics to the main loss plot
        stats_text = (f'Training Duration: {stats["training_duration_hours"]:.1f} hours\n'
                      f'Total Iterations: {stats["total_iterations"]:,}\n'
                      f'Final Val Loss: {stats["final_val_loss"]:.4f}')

        ax_loss.text(0.02, 0.98, stats_text, transform=ax_loss.transAxes,
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        

    def analyze_validation_loss(self, df, dataset_name):
        """Analyze validation loss trends and create visualizations."""
        val_data = df[df['val_loss'].notna()].copy()
        if val_data.empty:
            raise ValueError("No validation loss data found in logs")

        duration_hours = (val_data['timestamp'].max() - val_data['timestamp'].min()).total_seconds() / 3600

        stats = {
            'total_iterations': len(df),
            'training_duration_hours': duration_hours,
            'initial_val_loss': val_data['val_loss'].iloc[0],
            'final_val_loss': val_data['val_loss'].iloc[-1],
            'best_val_loss': val_data['val_loss'].min(),
            'worst_val_loss': val_data['val_loss'].max(),
            'mean_val_loss': val_data['val_loss'].mean(),
            'val_loss_std': val_data['val_loss'].std(),
            'convergence_iterations': val_data[val_data['val_loss'] == val_data['val_loss'].min()]['iteration'].iloc[0]
        }

        report = [
            "Validation Loss Analysis Report",
            "==============================\n",
            f"Dataset: {dataset_name}",
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nTraining Statistics:",
            f"Total Iterations: {stats['total_iterations']:,}",
            f"Training Duration: {stats['training_duration_hours']:.2f} hours",
            f"Iterations to Best Loss: {stats['convergence_iterations']:,}",
            f"\nValidation Loss Metrics:",
            f"Initial: {stats['initial_val_loss']:.6f}",
            f"Final: {stats['final_val_loss']:.6f}",
            f"Best: {stats['best_val_loss']:.6f}",
            f"Worst: {stats['worst_val_loss']:.6f}",
            f"Mean: {stats['mean_val_loss']:.6f}",
            f"Standard Deviation: {stats['val_loss_std']:.6f}",
            f"\nConvergence Analysis:",
            f"Time to Best Loss: {stats['convergence_iterations']/stats['total_iterations']*100:.1f}% of total iterations"
        ]

        if 'use_extra_cond' in df.columns:
            extra_cond_info = df.iloc[-1]
            report.extend([
                f"\nExtra Conditions:",
                f"Enabled: {extra_cond_info['use_extra_cond']}",
                f"Active Conditions: {extra_cond_info['active_conditions']}" if extra_cond_info['use_extra_cond'] else ""
            ])

        with open(os.path.join(self.output_dir, f'validation_report_{dataset_name.lower()}.txt'), 'w') as f:
            f.write('\n'.join(report))

        return stats, val_data

def main():
    parser = argparse.ArgumentParser(description="Analyze validation loss from training logs")
    parser.add_argument("--dataset", type=str, choices=['baseline', 'reflection', 'diffraction', 'combined'],
                        help="Dataset type to analyze")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Directory containing the training log files (e.g., training_metrics_*.jsonl)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where the analysis results (plots, reports) will be saved")
    args = parser.parse_args()

    # Dataset-specific names
    dataset_names = {
        'baseline': 'Baseline',
        'reflection': 'Reflection',
        'diffraction': 'Diffraction',
        'combined': 'Combined'
    }

    base_log_dir = args.log_dir
    base_output_dir = args.output_dir

    all_stats = {}
    all_dfs = {}

    # Dataset-Infos
    dataset_info = {
        'baseline': {
            'log_dir': f"{base_log_dir}/urban_sound_25k_baseline/Checkpoints/soundmap/256x256/glow_improved/building2soundmap/logs",
            'output_dir': f"{base_output_dir}/urban_sound_25k_baseline/analysis/validation_loss",
            'name': 'Baseline'
        },
        'reflection': {
            'log_dir': f"{base_log_dir}/urban_sound_25k_reflection/Checkpoints/soundmap/256x256/glow_improved/building2soundmap/logs",
            'output_dir': f"{base_output_dir}/urban_sound_25k_reflection/analysis/validation_loss",
            'name': 'Reflection'
        },
        'diffraction': {
            'log_dir': f"{base_log_dir}/urban_sound_25k_diffraction/Checkpoints/soundmap/256x256/glow_improved/building2soundmap/logs",
            'output_dir': f"{base_output_dir}/urban_sound_25k_diffraction/analysis/validation_loss",
            'name': 'Diffraction'
        },
        'combined': {
            'log_dir': f"{base_log_dir}/urban_sound_25k_combined/Checkpoints/soundmap/256x256/glow_improved/building2soundmap/logs",
            'output_dir': f"{base_output_dir}/urban_sound_25k_combined/analysis/validation_loss",
            'name': 'Combined'
        }
    }

    # Erstelle eine Figure für das 2x2-Grid
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)

    # Iteriere über die Datensätze und sammle die Ergebnisse
    plot_index = 0
    for dataset_key, dataset in dataset_info.items():
        if args.dataset is None or dataset_key == args.dataset:
            analyzer = ValidationLossAnalyzer(dataset['log_dir'], dataset['output_dir'])
            dataset_name = dataset_names.get(dataset_key, "Unknown Dataset")

            print(f"\nAnalyzing validation loss for {dataset_name} dataset")
            print(f"Log directory: {dataset['log_dir']}")
            print(f"Output directory: {dataset['output_dir']}")

            try:
                df = analyzer.parse_log_files()
                stats, val_data = analyzer.analyze_validation_loss(df, dataset_name)

                print("\nAnalysis complete!")
                print(f"Results saved to: {dataset['output_dir']}")
                print(f"\nKey Statistics:")
                print(f"Best validation loss: {stats['best_val_loss']:.6f}")
                print(f"Final validation loss: {stats['final_val_loss']:.6f}")
                print(f"Training duration: {stats['training_duration_hours']:.2f} hours")
                print(f"Iterations to best loss: {stats['convergence_iterations']:,}")

                all_stats[dataset_key] = stats
                all_dfs[dataset_key] = df

                # Füge die Plots dem entsprechenden Subplot im Grid hinzu
                ax_loss = fig.add_subplot(gs[plot_index // 2, plot_index % 2])
                analyzer.create_scientific_plot(val_data, stats, dataset_name, ax_loss)

                plot_index += 1

            except Exception as e:
                print(f"\nError during analysis for {dataset_name}: {str(e)}")

    # Speichere den Gesamtplot mit 4 Subplots
    fig.tight_layout()
    plt.savefig(os.path.join(base_output_dir, 'combined_validation_loss_plots.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(base_output_dir, 'combined_validation_loss_plots.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()