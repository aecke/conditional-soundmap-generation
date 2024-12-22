import time
import json
import sys
import os
from datetime import datetime
import torch
from helper.paths import compute_paths, make_dir_if_not_exists

class TrainingLogger:
    def __init__(self, args, params):
        """
        Initialize training logger focused on comprehensive data collection.
        
        Args:
            args: Command line arguments
            params: Model and training parameters
        """
        # Paths setup
        paths = compute_paths(args, params)
        self.checkpoints_path = paths["checkpoints_path"]
        self.log_dir = os.path.join(self.checkpoints_path, "logs")
        make_dir_if_not_exists(self.log_dir)
        
        # Basic info
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = args.model
        self.start_time = time.time()
        self.batch_size = params['batch_size']
        
        # Training parameters
        self.params = params
        self.args = args
        
        # Initialize counters and best values
        self.best_val_loss = float('inf')
        self.steps_without_improvement = 0
        self.total_training_steps = 0
        self.nan_inf_counts = 0
        
        # Extra Conditions Configuration
        self.use_extra_cond = any([
            args.use_temperature,
            args.use_humidity, 
            args.use_db
        ])
        self.active_conditions = []
        if args.use_temperature:
            self.active_conditions.append("temperature")
        if args.use_humidity:
            self.active_conditions.append("humidity")
        if args.use_db:
            self.active_conditions.append("db")

        # Initialize log files with clear suffixes
        self.log_file = os.path.join(self.log_dir, f'training_metrics_{self.timestamp}.jsonl')
        self.val_file = os.path.join(self.log_dir, f'validation_metrics_{self.timestamp}.jsonl')
        self.summary_file = os.path.join(self.log_dir, f'training_summary_{self.timestamp}.txt')
        self.system_metrics_file = os.path.join(self.log_dir, f'system_metrics_{self.timestamp}.jsonl')
        self.config_file = os.path.join(self.log_dir, f'experiment_config_{self.timestamp}.json')
        
        # Log initial configuration
        self._log_hyperparameters()

    def _log_hyperparameters(self):
        """Log complete experiment configuration and setup."""
        hyperparameters = {
            'timestamp': self.timestamp,
            'model_configuration': {
                'model_name': self.model_name,
                'n_flow': self.params['n_flow'],
                'n_block': self.params['n_block'],
                'img_size': self.params['img_size'],
                'channels': self.params['channels'],
                'batch_size': self.batch_size,
                'learning_rate': self.params['lr'],
                'temperature': self.params['temperature']
            },
            'training_configuration': {
                'direction': self.args.direction,
                'dataset': self.args.dataset,
                'reg_factor': self.args.reg_factor,
                'grad_checkpoint': self.args.grad_checkpoint,
                'do_lu': self.args.do_lu,
                'max_iterations': self.params['iter'],
                'checkpoint_freq': self.params['checkpoint_freq'],
                'sample_freq': self.params['sample_freq'],
                'val_freq': self.params['val_freq']
            },
            'extra_conditioning': {
                'enabled': self.use_extra_cond,
                'active_conditions': self.active_conditions
            },
            'system_info': {
                'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'cpu',
                'torch_version': torch.__version__,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'python_version': sys.version,
            },
            'data_paths': {
                'checkpoints': self.checkpoints_path,
                'logs': self.log_dir,
                'samples': self.params['samples_path']
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(hyperparameters, f, indent=4)

    def log_iteration(self, optim_step, metrics, iteration_time):
        """Log comprehensive metrics for each iteration."""
        self.total_training_steps += 1
        
        # System metrics
        memory = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
        memory_reserved = torch.cuda.memory_reserved()/1e9 if torch.cuda.is_available() else 0
        
        # Check for NaN/Inf values
        if any(torch.isnan(torch.tensor([v])) if v is not None else False for v in metrics.values()):
            self.nan_inf_counts += 1
        
        # Training metrics with timestamp
        training_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'iteration': optim_step,
            'epoch_approx': optim_step // self.batch_size,
            'train_metrics': {
                'loss': metrics.get('loss'),
                'loss_left': metrics.get('loss_left'),
                'loss_right': metrics.get('loss_right'),
                'learning_rate': metrics.get('learning_rate'),
            },
            'validation_metrics': {
                'val_loss': metrics.get('val_loss'),
                'steps_without_improvement': self.steps_without_improvement,
                'best_val_loss': self.best_val_loss
            } if 'val_loss' in metrics else None,
            'extra_conditions': {
                'enabled': self.use_extra_cond,
                'active': self.active_conditions
            } if self.use_extra_cond else None
        }

        # System metrics in separate file
        system_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'iteration': optim_step,
            'system_metrics': {
                'iteration_time': iteration_time,
                'memory_allocated_gb': float(memory),
                'memory_reserved_gb': float(memory_reserved),
                'memory_peaked_gb': float(torch.cuda.max_memory_allocated()/1e9) if torch.cuda.is_available() else 0,
                'cuda_memory_allocated_bytes': int(torch.cuda.memory_allocated()),
                'cuda_max_memory_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                'cuda_memory_cached_bytes': int(torch.cuda.memory_reserved())
            }
        }
        
        # Write to respective log files
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(training_entry) + '\n')
            
        with open(self.system_metrics_file, 'a') as f:
            f.write(json.dumps(system_entry) + '\n')

    def log_validation(self, val_metrics, iteration):
        """Log detailed validation metrics and track improvement."""
        current_val_loss = val_metrics.get('val_loss')
        
        # Update best validation loss and steps without improvement
        if current_val_loss is not None:
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1

        validation_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'iteration': iteration,
            'epoch_approx': iteration // self.batch_size,
            'validation_metrics': {
                'val_loss': current_val_loss,
                'val_loss_left': val_metrics.get('val_loss_left'),
                'val_loss_right': val_metrics.get('val_loss_right'),
                'best_val_loss_so_far': self.best_val_loss,
                'steps_without_improvement': self.steps_without_improvement
            },
            'extra_conditions': {
                'enabled': self.use_extra_cond,
                'active': self.active_conditions
            } if self.use_extra_cond else None
        }
        
        with open(self.val_file, 'a') as f:
            f.write(json.dumps(validation_entry) + '\n')

    def write_summary(self):
        """Write final training summary with comprehensive statistics."""
        duration = time.time() - self.start_time
        
        summary = f"""Training Summary
=================
Model: {self.model_name}
Run timestamp: {self.timestamp}

Training Statistics:
- Total training time: {duration/3600:.2f} hours
- Total training steps: {self.total_training_steps}
- Best validation loss: {self.best_val_loss:.6f}
- Final steps without improvement: {self.steps_without_improvement}
- NaN/Inf occurrences: {self.nan_inf_counts}

Extra Conditioning:
- Enabled: {self.use_extra_cond}
- Active conditions: {', '.join(self.active_conditions) if self.active_conditions else 'None'}

Training Configuration:
- Batch size: {self.batch_size}
- Learning rate: {self.params['lr']}
- Model blocks: {self.params['n_block']}
- Flow steps: {self.params['n_flow']}
- Image size: {self.params['img_size']}
- Direction: {self.args.direction}
- Dataset: {self.args.dataset}

Log Files (JSONL format):
- Training metrics: {self.log_file}
- Validation metrics: {self.val_file}
- System metrics: {self.system_metrics_file}
- Full configuration: {self.config_file}

Note: All metrics are stored in JSONL format for easy post-processing and visualization.
Each log entry contains detailed timestamps and comprehensive metrics.
"""
        
        with open(self.summary_file, 'w') as f:
            f.write(summary)