import torch
import os
import argparse
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms, utils
from tqdm import tqdm
import helper
import models
from globals import device
from sklearn.preprocessing import MinMaxScaler

def parse_args():
    parser = argparse.ArgumentParser(description="Generate soundmaps from test buildings")
    parser.add_argument("--use_temperature", action="store_true", help="Use temperature condition")
    parser.add_argument("--use_humidity", action="store_true", help="Use humidity condition")
    parser.add_argument("--use_db", action="store_true", help="Use dB condition")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to specific checkpoint")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to specific dataset")
    parser.add_argument("--optim_step", type=int, default=1000000, help="Optimization step to load")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args_cmd = parse_args()

    # CUDA Check
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")

    # Paths - allow override through command line
    CHECKPOINT_PATH = args_cmd.checkpoint_path or "E:/ba_ergebnisse/urban_sound_25k_reflection/Checkpoints/soundmap/256x256/glow_improved/building2soundmap"
    TEST_DATA_PATH = args_cmd.dataset_path or "E:/Schallsimulationsdaten/urban_sound_25k_reflection/test"
    OUTPUT_PATH = "E:/ba_ergebnisse/urban_sound_25k_reflection/evaluation_results/pred"

    # Create output directory
    helper.make_dir_if_not_exists(OUTPUT_PATH)
    print(f"Output directory ready: {OUTPUT_PATH}")

    # Load base parameters
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
            # Use conditions from command line arguments
            self.use_temperature = args_cmd.use_temperature
            self.use_humidity = args_cmd.use_humidity
            self.use_db = args_cmd.use_db
            self.exp = False
            self.last_optim_step = args_cmd.optim_step

    args = Args()

    # Print configuration
    print("\nModel Configuration:")
    print(f"Number of flow steps: {params['n_flow']}")
    print(f"Number of blocks: {params['n_block']}")
    print(f"Image size: {params['img_size']}")
    print(f"Temperature: {params['temperature']}")
    print("\nExtra Conditions:")
    print(f"Using temperature: {args.use_temperature}")
    print(f"Using humidity: {args.use_humidity}")
    print(f"Using dB: {args.use_db}")

    # Initialize model with error handling
    try:
        print("\nInitializing model...")
        model = models.init_model(args, params)
        
        print(f"Loading checkpoint from step {args.last_optim_step}...")
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
        
        required_columns = ['osm']
        if args.use_db:
            required_columns.append('db')
        if args.use_temperature:
            required_columns.append('temperature')
        if args.use_humidity:
            required_columns.append('humidity')
            
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

    # Setup scalers for numerical conditions
    scalers = {}
    if args.use_db:
        scalers['db'] = MinMaxScaler()
        db_range = np.linspace(0, 100, 1000).reshape(-1, 1)
        scalers['db'].fit(db_range)
        print("\nDB Scaler Range:")
        print(f"Min: {scalers['db'].data_min_[0]:.2f}")
        print(f"Max: {scalers['db'].data_max_[0]:.2f}")

    if args.use_temperature:
        scalers['temperature'] = MinMaxScaler()
        temp_range = np.linspace(-20, 40, 1000).reshape(-1, 1)  # Angepasster Temperaturbereich
        scalers['temperature'].fit(temp_range)
        
    if args.use_humidity:
        scalers['humidity'] = MinMaxScaler()
        humidity_range = np.linspace(0, 100, 1000).reshape(-1, 1)
        scalers['humidity'].fit(humidity_range)

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
                
                # Process extra conditions
                extra_cond_values = []
                
                if args.use_db:
                    try:
                        db_value = eval(row['db'])['lwd500']
                        normalized_db = scalers['db'].transform([[db_value]])[0][0]
                        extra_cond_values.append(normalized_db)
                    except (KeyError, SyntaxError) as e:
                        raise ValueError(f"Invalid dB value format: {e}")
                        
                if args.use_temperature:
                    temp_value = float(row['temperature'])
                    normalized_temp = scalers['temperature'].transform([[temp_value]])[0][0]
                    extra_cond_values.append(normalized_temp)
                    
                if args.use_humidity:
                    humidity_value = float(row['humidity'])
                    normalized_humidity = scalers['humidity'].transform([[humidity_value]])[0][0]
                    extra_cond_values.append(normalized_humidity)
                
                if extra_cond_values:
                    batch['extra_cond'] = torch.tensor(extra_cond_values, dtype=torch.float32).to(device)
                
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

if __name__ == "__main__":
    main()