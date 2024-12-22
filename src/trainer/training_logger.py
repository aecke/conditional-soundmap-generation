import json
import os
from datetime import datetime
import torch
from helper.paths import compute_paths, make_dir_if_not_exists

class TrainingLogger:
    def __init__(self, args, params):
        """
        Initialize the training logger.
        
        Args:
            args: Command line arguments
            params: Model and training parameters
        """
        # Get paths from existing path computation
        paths = compute_paths(args, params)
        self.checkpoints_path = paths["checkpoints_path"]
        self.log_dir = os.path.join(self.checkpoints_path, "logs")
        make_dir_if_not_exists(self.log_dir)
        
        # Create timestamp for unique run identification
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = args.model
        
        # Initialize metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'loss_left': [],
            'loss_right': [], 
            'learning_rate': [],
            'iteration': [],
            'iteration_time': [],
            'memory_usage': [],
        }
        
        # Initialize log files using existing path structure
        self.log_file = os.path.join(self.log_dir, f'training_log_{self.timestamp}.json')
        self.summary_file = os.path.join(self.log_dir, f'training_summary_{self.timestamp}.txt')
        
        # Log initial configuration
        self.log_hyperparameters(args, params)

    def log_hyperparameters(self, args, params):
        """Log training hyperparameters at the start of training."""
        hyperparameters = {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'args': vars(args),
            'params': params,
            'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'cpu'
        }
        
        with open(self.log_file, 'w') as f:
            json.dump({'hyperparameters': hyperparameters}, f, indent=4)

    def log_iteration(self, optim_step, metrics, time_taken):
        """
        Log metrics for a single training iteration.
        
        Args:
            optim_step (int): Current optimization step
            metrics (dict): Dictionary containing current metrics
            time_taken (float): Time taken for iteration
        """
        memory = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
        
        # Update internal metrics
        self.metrics['iteration'].append(optim_step)
        self.metrics['train_loss'].append(metrics.get('loss', None))
        self.metrics['val_loss'].append(metrics.get('val_loss', None))
        self.metrics['loss_left'].append(metrics.get('loss_left', None))
        self.metrics['loss_right'].append(metrics.get('loss_right', None))
        self.metrics['learning_rate'].append(metrics.get('learning_rate', None))
        self.metrics['iteration_time'].append(time_taken)
        self.metrics['memory_usage'].append(memory)

        # Append to log file
        with open(self.log_file, 'a') as f:
            log_entry = {
                'iteration': optim_step,
                'metrics': metrics,
                'time_taken': time_taken,
                'memory_used': memory
            }
            f.write(json.dumps(log_entry) + '\n')

    def write_summary(self):
        """Write a summary of the training run including best metrics."""
        best_metrics = self.get_best_metrics()
        avg_iteration_time = sum(self.metrics['iteration_time']) / len(self.metrics['iteration_time'])
        
        summary = f"""Training Summary
=================
Model: {self.model_name}
Run timestamp: {self.timestamp}
Total iterations: {len(self.metrics['iteration'])}

Best Metrics:
- Best training loss: {best_metrics['best_train_loss']:.6f}
- Best validation loss: {best_metrics['best_val_loss']:.6f}
- Best left loss: {best_metrics['best_loss_left']:.6f}
- Best right loss: {best_metrics['best_loss_right']:.6f}

Final Metrics:
- Final training loss: {best_metrics['final_train_loss']:.6f}
- Final validation loss: {best_metrics['final_val_loss']:.6f}

Performance:
- Average iteration time: {avg_iteration_time:.3f}s
- Peak memory usage: {max(self.metrics['memory_usage']):.2f}GB
"""
        
        with open(self.summary_file, 'w') as f:
            f.write(summary)

    def get_best_metrics(self):
        """Return dictionary of best metrics achieved during training."""
        return {
            'best_train_loss': min(x for x in self.metrics['train_loss'] if x is not None),
            'best_val_loss': min(x for x in self.metrics['val_loss'] if x is not None),
            'best_loss_left': min(x for x in self.metrics['loss_left'] if x is not None),
            'best_loss_right': min(x for x in self.metrics['loss_right'] if x is not None),
            'final_train_loss': next(x for x in reversed(self.metrics['train_loss']) if x is not None),
            'final_val_loss': next(x for x in reversed(self.metrics['val_loss']) if x is not None)
        }