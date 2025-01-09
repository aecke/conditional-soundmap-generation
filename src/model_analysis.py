import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import helper
import models
from globals import device
from sklearn.preprocessing import MinMaxScaler

class ModelAnalyzer:
    def __init__(self, checkpoint_path, test_path, output_path):
        self.checkpoint_path = checkpoint_path
        self.test_path = test_path
        self.output_path = output_path
        
        # Parameters
        self.params = helper.read_params("../params.json")["soundmap"]
        self.params.update({
            'n_samples': 1,
            'batch_size': 1,
            'img_size': [256, 256]
        })
        
        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        
    def calc_input_shapes(self, extra_cond_dim=32):
        """Calculate input shapes for the model based on extra conditions."""
        n_channels = self.params['channels']
        image_size = self.params['img_size']
        n_blocks = self.params['n_block']
        
        # Base input shapes ohne extra conditions
        base_shapes = calc_inp_shapes(n_channels, image_size, n_blocks, 'regular')
        
        # Add extra condition dimensions
        shapes_with_cond = []
        for shape in base_shapes:
            new_shape = list(shape)
            new_shape[0] += extra_cond_dim  # Erweitere Channel-Dimension
            shapes_with_cond.append(tuple(new_shape))
            
        return shapes_with_cond
            
    def setup_model(self, temperature=1.0, use_conditions=True):
        """Initialize model with specific configuration."""
        params = self.params  # Lokale Referenz für Zugriff in Args
        
        class Args:
            def __init__(self):
                self.dataset = 'soundmap'
                self.direction = 'building2soundmap'
                self.model = 'glow_improved'
                self.do_lu = True
                self.grad_checkpoint = False
                self.n_block = params['n_block']
                self.n_flow = params['n_flow']
                self.reg_factor = 0.0001
                self.use_bmaps = False
                self.use_temperature = use_conditions
                self.use_humidity = use_conditions
                self.use_db = use_conditions
                self.exp = False
                self.last_optim_step = 1000000
        
        args = Args()
        self.params['temperature'] = temperature
        
        # Initialisiere das Modell über das models interface
        model = models.init_model(args, self.params)
        checkpoint = helper.load_checkpoint(
            self.checkpoint_path,
            args.last_optim_step,
            model,
            None,
            resume_train=False
        )
        model = checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint
        model.eval()
        
        return model, args
        
        self.params['temperature'] = temperature
        
        model = models.init_model(Args(), self.params)
        checkpoint = helper.load_checkpoint(
            self.checkpoint_path,
            Args().last_optim_step,
            model,
            None,
            resume_train=False
        )
        model = checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint
        model.eval()
        
        return model, Args()
    
    def setup_scalers(self):
        """Setup scalers for conditions."""
        scalers = {}
        
        # DB Scaler (0-100)
        scalers['db'] = MinMaxScaler()
        scalers['db'].fit(np.linspace(0, 100, 1000).reshape(-1, 1))
        
        # Temperature Scaler (-20 to 40)
        scalers['temperature'] = MinMaxScaler()
        scalers['temperature'].fit(np.linspace(-20, 40, 1000).reshape(-1, 1))
        
        # Humidity Scaler (0-100)
        scalers['humidity'] = MinMaxScaler()
        scalers['humidity'].fit(np.linspace(0, 100, 1000).reshape(-1, 1))
        
        return scalers
    
    def generate_samples(self, building, conditions, n_variations=5):
        """Generate multiple samples for the same building with different parameters."""
        # Test different temperatures
        temperatures = [0.1, 0.5, 1.0]
        
        # Test with and without conditions
        configs = [
            ("with_conditions", True),
            ("without_conditions", False)
        ]
        
        results = []
        
        for temp in temperatures:
            for config_name, use_conditions in configs:
                print(f"\nGenerating samples with temperature {temp} and {config_name}")
                
                # Setup model
                model, args = self.setup_model(temperature=temp, use_conditions=use_conditions)
                
                # Generate multiple samples
                for i in range(n_variations):
                    batch = {'building': building}
                    
                    if use_conditions and conditions is not None:
                        batch['extra_cond'] = conditions
                    
                    with torch.no_grad():
                        samples = models.take_samples(
                            args,
                            self.params,
                            model,
                            batch['building'],
                            batch,
                            n_samples=1
                        )
                    
                    results.append({
                        'temperature': temp,
                        'config': config_name,
                        'variation': i,
                        'sample': samples[0]
                    })
        
        return results
    
    def analyze_single_case(self, index=0):
        """Analyze a single test case in detail."""
        # Load test data
        test_csv = pd.read_csv(os.path.join(self.test_path, "test.csv"))
        sample_row = test_csv.iloc[index]
        
        # Load building
        building_path = os.path.join(self.test_path, "buildings", os.path.basename(sample_row['osm']))
        transform = transforms.Compose([
            transforms.Resize(self.params['img_size']),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        
        building = transform(Image.open(building_path))
        if building.size(0) == 1:
            building = building.repeat(3, 1, 1)
        building = building.unsqueeze(0).to(device)
        
        # Load ground truth
        truth_path = os.path.join(self.test_path, sample_row.soundmap.replace("./", ""))
        truth_map = transform(Image.open(truth_path))
        if truth_map.size(0) == 1:
            truth_map = truth_map.repeat(3, 1, 1)
        
        # Prepare conditions
        scalers = self.setup_scalers()
        extra_cond_values = []
        
        if 'db' in sample_row:
            db_value = eval(sample_row['db'])['lwd500']
            normalized_db = scalers['db'].transform([[db_value]])[0][0]
            extra_cond_values.append(normalized_db)
            
        if 'temperature' in sample_row:
            temp_value = float(sample_row['temperature'])
            normalized_temp = scalers['temperature'].transform([[temp_value]])[0][0]
            extra_cond_values.append(normalized_temp)
            
        if 'humidity' in sample_row:
            humidity_value = float(sample_row['humidity'])
            normalized_humidity = scalers['humidity'].transform([[humidity_value]])[0][0]
            extra_cond_values.append(normalized_humidity)
        
        conditions = torch.tensor(extra_cond_values, dtype=torch.float32).to(device) if extra_cond_values else None
        
        # Generate samples
        results = self.generate_samples(building, conditions)
        
        # Create visualization
        n_rows = len(results) // 5 + (1 if len(results) % 5 != 0 else 0)
        fig, axes = plt.subplots(n_rows + 1, 5, figsize=(20, 4*n_rows + 4))
        
        # Plot ground truth and building in first row
        axes[0,0].imshow(building.cpu().squeeze().permute(1,2,0))
        axes[0,0].set_title("Input Building")
        axes[0,1].imshow(truth_map.squeeze().permute(1,2,0))
        axes[0,1].set_title("Ground Truth")
        for i in range(2, 5):
            axes[0,i].axis('off')
        
        # Plot generated samples
        for idx, result in enumerate(results):
            row = idx // 5 + 1
            col = idx % 5
            
            sample = result['sample'].cpu()
            axes[row,col].imshow(sample.squeeze().permute(1,2,0))
            axes[row,col].set_title(f"T={result['temperature']}\n{result['config']}\n#{result['variation']}")
        
        # Turn off remaining axes
        for row in range(1, n_rows + 1):
            for col in range((len(results) - (row-1)*5) % 5, 5):
                axes[row,col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f"analysis_sample_{index}.png"))
        plt.close()
        
        return building, truth_map, results

def main():
    # Paths
    CHECKPOINT_PATH = "E:/ba_ergebnisse/urban_sound_25k_combined/Checkpoints/soundmap/256x256/glow_improved/building2soundmap"
    TEST_PATH = "E:/Schallsimulationsdaten/urban_sound_25k_combined/test"
    OUTPUT_PATH = "E:/ba_ergebnisse/urban_sound_25k_combined/analysis_results"
    
    analyzer = ModelAnalyzer(CHECKPOINT_PATH, TEST_PATH, OUTPUT_PATH)
    
    # Analyze first 5 samples
    for i in range(5):
        print(f"\nAnalyzing sample {i}...")
        building, truth, results = analyzer.analyze_single_case(i)
        print(f"Generated {len(results)} variations")

if __name__ == "__main__":
    main()