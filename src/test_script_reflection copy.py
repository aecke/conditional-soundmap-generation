import torch
import os
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms, utils
from tqdm import tqdm
import helper
import models
from globals import device
from sklearn.preprocessing import MinMaxScaler

# CUDA Check
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, using CPU")

# Paths
CHECKPOINT_PATH = "E:/ba_ergebnisse/urban_sound_25k_reflection/Checkpoints/soundmap/256x256/glow_improved/building2soundmap"
TEST_DATA_PATH = "E:/Schallsimulationsdaten/urban_sound_25k_reflection/test"
OUTPUT_PATH = "E:/ba_ergebnisse/urban_sound_25k_reflection/evaluation_test_dataset_results/pred"

# Create output directory
helper.make_dir_if_not_exists(OUTPUT_PATH)
print(f"Output directory ready: {OUTPUT_PATH}")

# Load base parameters from params.json
params = helper.read_params("../params.json")["soundmap"]

# Override only necessary parameters for testing
params.update({
    'n_samples': 1,
    'batch_size': 1,
    'temperature': 1,
})

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
        self.use_temperature = False
        self.use_humidity = False  
        self.use_db = True
        self.exp = False
        self.last_optim_step = 1000000

args = Args()

# Print configuration
print("\nModel Configuration:")
print(f"Number of flow steps: {params['n_flow']}")
print(f"Number of blocks: {params['n_block']}")
print(f"Image size: {params['img_size']}")
print(f"Temperature: {params['temperature']}")
print(f"Using dB condition: {args.use_db}")

# Initialize model with error handling
try:
    print("\nInitializing model...")
    model = models.init_model(args, params)
    
    print("Loading checkpoint...")
    checkpoint = helper.load_checkpoint(CHECKPOINT_PATH, args.last_optim_step, model, None, resume_train=False)
    if isinstance(checkpoint, tuple):
        model = checkpoint[0]
    else:
        model = checkpoint
    
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error during model initialization: {str(e)}")
    raise

# Load and validate test dataset
try:
    print("\nLoading test dataset...")
    test_csv = pd.read_csv(os.path.join(TEST_DATA_PATH, "test.csv"))
    required_columns = ['osm', 'db']
    missing_columns = [col for col in required_columns if col not in test_csv.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in test.csv: {missing_columns}")
    print(f"Loaded {len(test_csv)} test samples")
except Exception as e:
    print(f"Error loading test dataset: {str(e)}")
    raise

# Setup image transform
transform = transforms.Compose([
    transforms.Resize(params['img_size']),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Setup dB scaler with extended range
db_scaler = MinMaxScaler()
if args.use_db:
    db_range = np.linspace(0, 100, 1000).reshape(-1, 1)
    db_scaler.fit(db_range)
    print("\nDB Scaler Range:")
    print(f"Min: {db_scaler.data_min_[0]:.2f}")
    print(f"Max: {db_scaler.data_max_[0]:.2f}")

# Processing statistics
total_samples = len(test_csv)
successful = 0
failed = 0
error_log = []

# Main processing loop
print("\nStarting prediction generation...")
with torch.no_grad():
    for idx, row in tqdm(test_csv.iterrows(), total=total_samples, desc="Generating predictions"):
        try:
            # Prepare batch dictionary
            batch = {}
            
            # Load and validate building image
            building_filename = os.path.basename(row['osm'])
            building_path = os.path.join(TEST_DATA_PATH, "buildings", building_filename)
            
            if not os.path.exists(building_path):
                raise FileNotFoundError(f"Building file not found: {building_path}")
                
            # Process building image
            building = transform(Image.open(building_path))
            if building.size(0) == 1:
                building = building.repeat(3, 1, 1)
            building = building.unsqueeze(0).to(device)
            batch['building'] = building
            
            # Process dB value if enabled
            if args.use_db:
                try:
                    db_value = eval(row['db'])['lwd500']
                    normalized_db = db_scaler.transform([[db_value]])[0][0]
                    batch['extra_cond'] = torch.tensor([normalized_db], dtype=torch.float32).to(device)
                except (KeyError, SyntaxError) as e:
                    raise ValueError(f"Invalid dB value format: {e}")
            
            # Generate prediction
            sampled_images = models.take_samples(
                args,
                params,
                model,
                batch['building'],
                batch,
                n_samples=1
            )
            
            # Validate generated image
            if torch.isnan(sampled_images).any():
                raise ValueError("Generated image contains NaN values")
            
            # Save the generated image
            utils.save_image(
                sampled_images[0],
                os.path.join(OUTPUT_PATH, f"y_{idx}.png"),
                normalize=True
            )
            
            successful += 1
            
            # Memory management
            if idx % 50 == 0:
                torch.cuda.empty_cache()
            
        except Exception as e:
            failed += 1
            error_msg = f"Error processing sample {idx}: {str(e)}"
            error_log.append(error_msg)
            print(f"\n{error_msg}")
            continue

# Print final statistics
print("\nProcessing complete:")
print(f"Total samples: {total_samples}")
print(f"Successfully processed: {successful}")
print(f"Failed: {failed}")
print(f"Success rate: {(successful/total_samples)*100:.2f}%")

# Save error log
if error_log:
    error_log_path = os.path.join(OUTPUT_PATH, "error_log.txt")
    with open(error_log_path, 'w') as f:
        f.write("\n".join(error_log))
    print(f"\nError log saved to: {error_log_path}")

print(f"\nResults saved to: {OUTPUT_PATH}")